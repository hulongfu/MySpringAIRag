package com.myspringairag.service;

import com.myspringairag.controller.SseController;
import com.myspringairag.model.Document;
import com.myspringairag.repository.DocumentRepository;
import lombok.extern.slf4j.Slf4j;
import org.springframework.ai.chat.client.ChatClient;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.web.multipart.MultipartFile;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;
import java.util.stream.Collectors;

@Slf4j
@Service
public class RagService {
    
    private final DocumentParserService parserService;
    private final EmbeddingService embeddingService;
    private final JVectorService jVectorService;
    private final DocumentRepository documentRepository;
    private final ChatClient chatClient;
    private final QueryRewriteService queryRewriteService;
    private final HierarchicalChunkingService hierarchicalChunkingService;
    private final SseController sseController;
    private final com.myspringairag.service.QueryTokenizationService tokenizationService;
    
    // 使用 ThreadLocal 存储当前任务的 taskId
    private static final ThreadLocal<String> currentTaskId = new ThreadLocal<>();
    
    @Autowired(required = false)
    private ParallelVectorRetrievalService parallelRetrievalService;
    
    @Autowired(required = false)
    private BgeReranker reranker;
    
    @Value("${jvector.top-k}")
    private int topK;
    
    @Value("${jvector.rerank-top-k:20}")
    private int rerankTopK;
    
    @Value("${app.use-reranking:true}")
    private boolean useReranking;
    
    @Value("${app.use-parallel-retrieval:false}")
    private boolean useParallelRetrieval;
    
    @Value("${app.use-semantic-chunking:false}")
    private boolean useSemanticChunking;
    
    public RagService(
            DocumentParserService parserService,
            EmbeddingService embeddingService,
            JVectorService jVectorService,
            DocumentRepository documentRepository,
            ChatClient.Builder chatClientBuilder,
            QueryRewriteService queryRewriteService,
            HierarchicalChunkingService hierarchicalChunkingService,
            SseController sseController,
            com.myspringairag.service.QueryTokenizationService tokenizationService) {
        this.parserService = parserService;
        this.embeddingService = embeddingService;
        this.jVectorService = jVectorService;
        this.documentRepository = documentRepository;
        this.chatClient = chatClientBuilder.build();
        this.queryRewriteService = queryRewriteService;
        this.hierarchicalChunkingService = hierarchicalChunkingService;
        this.sseController = sseController;
        this.tokenizationService = tokenizationService;
    }
    
    /**
     * 上传文件并建立索引
     */
    @Transactional
    public void uploadDocument(MultipartFile file) {
        try {
            log.info("Uploading document: {}", file.getOriginalFilename());
            
            // 1. 解析文件
            String text = parserService.parseFile(file);
            
            if (text == null || text.trim().isEmpty()) {
                throw new IllegalArgumentException("Document is empty or could not be parsed");
            }
            
            processAndIndex(text, file.getOriginalFilename(), file.getSize(), file.getContentType());
            
        } catch (Exception e) {
            log.error("Failed to upload document: {}", file.getOriginalFilename(), e);
            throw new RuntimeException("Document upload failed: " + e.getMessage(), e);
        }
    }
    
    /**
     * 设置当前任务ID（由 AsyncUploadService 调用）
     */
    public static void setCurrentTaskId(String taskId) {
        currentTaskId.set(taskId);
    }
    
    /**
     * 清除当前任务ID
     */
    public static void clearCurrentTaskId() {
        currentTaskId.remove();
    }
    
    /**
     * 获取当前任务ID
     */
    private static String getCurrentTaskId() {
        return currentTaskId.get();
    }
    
    /**
     * 从文件路径上传并建立索引（用于异步处理）
     */
    @Transactional
    public void uploadDocumentFromPath(Path filePath, String filename) {
        String taskId = getCurrentTaskId();
        
        try {
            log.info("Uploading document from path: {}", filePath);
            
            // 0. 检查文件是否已存在
            List<Document> existingDocs = documentRepository.findByFilename(filename);
            if (!existingDocs.isEmpty()) {
                throw new IllegalArgumentException(
                    "文件已存在：" + filename + "。请先删除旧文档后再上传。"
                );
            }
            
            // 10% - 开始读取文件
            if (taskId != null) {
                sseController.notifyProgress(taskId, 10, "正在读取文件...");
            }
            
            // 1. 解析文件
            String text = parserService.parseFileFromPath(filePath);
            
            if (text == null || text.trim().isEmpty()) {
                throw new IllegalArgumentException("Document is empty or could not be parsed");
            }
            
            // 30% - 文件解析完成
            if (taskId != null) {
                sseController.notifyProgress(taskId, 30, "正在解析文档...");
            }
            
            long fileSize = Files.size(filePath);
            String mimeType = Files.probeContentType(filePath);
            
            processAndIndex(text, filename, fileSize, mimeType);
            
        } catch (Exception e) {
            log.error("Failed to upload document from path: {}", filePath, e);
            throw new RuntimeException("Document upload failed: " + e.getMessage(), e);
        }
    }
    
    /**
     * 处理文本并建立索引（公共逻辑）
     */
    private void processAndIndex(String text, String filename, long fileSize, String mimeType) {
        String taskId = getCurrentTaskId();
        
        // 2. 分割文本（使用层级分块）
        List<String> chunks;
        List<String> parentContents;
        
        if (useSemanticChunking) {
            // 使用层级分块
            log.info("Using hierarchical chunking (Parent-Child strategy)");
            List<org.springframework.ai.document.Document> hierarchicalDocs = 
                hierarchicalChunkingService.hierarchicalChunk(text, filename);
            
            // 提取小块和大块
            chunks = new ArrayList<>();
            parentContents = new ArrayList<>();
            for (org.springframework.ai.document.Document doc : hierarchicalDocs) {
                chunks.add((String) doc.getMetadata().get("content"));
                parentContents.add((String) doc.getMetadata().get("parentContent"));
            }
        } else {
            // 使用原有分块方式
            chunks = parserService.splitIntoChunks(text);
            parentContents = null;  // 不使用层级分块时，parentContent为null
        }
        
        if (chunks.isEmpty()) {
            throw new IllegalArgumentException("No valid chunks generated from document");
        }
        
        // 60% - 分块完成
        if (taskId != null) {
            sseController.notifyProgress(taskId, 60, "正在生成向量...");
        }
        
        // 3. 批量生成向量并收集docIds（优化：先保存文档，再批量生成向量）
        List<Long> docIds = new ArrayList<>();
        List<String> chunksToEmbed = new ArrayList<>();
        
        // 第1步：保存所有文档到数据库
        for (int i = 0; i < chunks.size(); i++) {
            String chunk = chunks.get(i);
            
            // 创建文档记录
            Document doc = new Document();
            doc.setFilename(filename);
            doc.setContent(chunk);
            // 设置parentContent（如果使用层级分块）
            if (parentContents != null && i < parentContents.size()) {
                doc.setParentContent(parentContents.get(i));
            }
            doc.setChunkIndex(i);
            doc.setTotalChunks(chunks.size());
            doc.setFileSize(fileSize);
            doc.setMimeType(mimeType);
            
            Document savedDoc = documentRepository.save(doc);
            docIds.add(savedDoc.getId());
            chunksToEmbed.add(chunk);
            
            log.debug("Saved chunk {}/{} for {}", i + 1, chunks.size(), filename);
        }
        
        // 65% - 文档保存完成
        if (taskId != null) {
            sseController.notifyProgress(taskId, 65, "正在批量生成向量...");
        }
        
        // 第2步：批量生成向量（并行优化）
        List<float[]> vectors = embeddingService.embedBatch(chunksToEmbed);
        
        // 70% - 向量化完成
        if (taskId != null) {
            sseController.notifyProgress(taskId, 70, "正在批量存储向量...");
        }
        
        // 4. 批量添加到JVector索引（不重建索引）
        jVectorService.addVectorsBatch(docIds, vectors);
        
        // 80% - 向量存储完成
        if (taskId != null) {
            sseController.notifyProgress(taskId, 80, "正在构建向量索引...");
        }
        
        // 5. 一次性重建索引（性能优化关键！）
        jVectorService.rebuildIndex();
        
        // 90% - 索引构建完成
        if (taskId != null) {
            sseController.notifyProgress(taskId, 90, "索引构建完成...");
        }
        
        log.info("Successfully uploaded and indexed document: {} ({} chunks)", filename, chunks.size());
    }
    
    /**
     * 基于知识库回答问题（查询转换 + 向量检索 + BGE重排序）
     */
    public String answerQuestion(String question) {
        long startTime = System.currentTimeMillis();
        try {
            log.info("Answering question: {}", question);
            
            // 0. 查询重写（优化为更适合向量检索的格式）
            String rewrittenQuery = queryRewriteService.rewrite(question);
                
            // 1. 向量检索
            long vectorSearchStart = System.currentTimeMillis();
            int searchTopK = useReranking ? rerankTopK : topK;
            
            List<Long> vectorResults;
            List<com.myspringairag.model.ScoredDocument> scoredDocs = null;  // 保存以便后续使用
            
            // 如果启用并行检索，使用多查询变体
            if (useParallelRetrieval && parallelRetrievalService != null) {
                log.info("Using parallel multi-query retrieval");
                scoredDocs = parallelRetrievalService.parallelSearch(rewrittenQuery, searchTopK);
                
                // 打印每个文档的相似度分数
                log.info("=== Vector Search Results with Scores ===");
                for (int i = 0; i < scoredDocs.size(); i++) {
                    com.myspringairag.model.ScoredDocument sd = scoredDocs.get(i);
                    log.info("Rank {}: docId={}, originalScore={}, rrfScore={}, finalScore={}",
                        i+1, sd.getId(), sd.getOriginalScore(), sd.getRrfScore(), sd.getFinalScore());
                }
                log.info("===========================================");
                
                // 提取docId列表
                vectorResults = scoredDocs.stream()
                    .map(com.myspringairag.model.ScoredDocument::getId)
                    .collect(Collectors.toList());
                
                log.info("Parallel retrieval returned {} results (took: {}ms)", 
                    vectorResults.size(), System.currentTimeMillis() - vectorSearchStart);
            } else {
                // 传统单查询检索
                
                float[] queryVector = embeddingService.embed(rewrittenQuery);
                
                // 使用带分数的搜索
                Map<Long, Float> docScores = jVectorService.searchWithScores(queryVector, searchTopK);
                
                // 转换为 ScoredDocument 列表（保持统一格式）
                scoredDocs = docScores.entrySet().stream()
                    .map(entry -> new com.myspringairag.model.ScoredDocument(
                        entry.getKey(),
                        null,  // content 不需要
                        entry.getValue(),  // originalScore
                        entry.getValue(),  // rrfScore (单查询时等于originalScore)
                        entry.getValue()   // finalScore
                    ))
                    .collect(Collectors.toList());
                
                // 提取docId列表
                vectorResults = scoredDocs.stream()
                    .map(com.myspringairag.model.ScoredDocument::getId)
                    .collect(Collectors.toList());
            }
                
            if (vectorResults.isEmpty()) {
                return "抱歉，我在知识库中没有找到相关的信息来回答您的问题。";
            }
            
            // 提取核心关键词
            List<String> coreKeywords = tokenizationService.extractCoreKeywords(rewrittenQuery);
                
            List<Document> finalDocs;
                
            // 2. 如果启用重排序，使用 BgeReranker 精排
            if (useReranking && reranker != null) {
                log.info("Using reranker to refine {} candidates", vectorResults.size());
                            
                // 【修改】获取所有候选文档内容（不限制数量）
                List<Document> candidateDocs = documentRepository.findByIds(vectorResults);
                
                // 如果是并行检索，从 ScoredDocument 复制分数
                if (scoredDocs != null) {
                    copyScoresToDocuments(candidateDocs, scoredDocs);
                }
                
                // 根据关键词调整分数并排序
                if (!coreKeywords.isEmpty()) {
                    candidateDocs = adjustScoreByKeywords(candidateDocs, coreKeywords);
                    
                    if (candidateDocs.isEmpty()) {
                        return "抱歉，我在知识库中没有找到包含关键信息（" + String.join(", ", coreKeywords) + "）的内容。";
                    }
                    
                    // 【新增】过滤掉分数为0的文档
                    candidateDocs = candidateDocs.stream()
                        .filter(doc -> doc.getSimilarityScore() != null && doc.getSimilarityScore() > 0)
                        .collect(Collectors.toList());
                    
                    if (candidateDocs.isEmpty()) {
                        return "抱歉，我在知识库中没有找到包含关键信息（" + String.join(", ", coreKeywords) + "）的内容。";
                    }
                }
                            
                // 打印候选文档信息
                log.info("=== Candidate Documents ===");
                for (int i = 0; i < candidateDocs.size(); i++) {
                    Document doc = candidateDocs.get(i);
                    String contentPreview = doc.getParentContent() != null ? 
                        doc.getParentContent() : doc.getContent();
                    log.info("Candidate {}: id={}, filename={}, chunk={}/{}, content_preview={}, similarityScore={}",
                        i+1, doc.getId(), doc.getFilename(), 
                        doc.getChunkIndex()+1, doc.getTotalChunks(),
                        contentPreview.substring(0, Math.min(100, contentPreview.length())), doc.getSimilarityScore());
                }
                log.info("===========================");
                            
                // 构建重排序输入（使用parentContent如果存在）
                String[] passages = candidateDocs.stream()
                    .map(doc -> doc.getParentContent() != null ? doc.getParentContent() : doc.getContent())
                    .toArray(String[]::new);
                            
                // 执行重排序
                Map<String, Float> rankedScores = reranker.rerankBatch(question, passages);
                            
                // 取 top-K
                List<Document> finalCandidateDocs = candidateDocs;
                finalDocs = rankedScores.entrySet().stream()
                    .limit(topK)
                    .map(entry -> findDocByContent(finalCandidateDocs, entry.getKey()))
                    .filter(Objects::nonNull)
                    .collect(Collectors.toList());
                            
                log.info("After reranking: {} documents selected", finalDocs.size());
            } else {
                // 不使用重排序，直接使用向量检索结果
                // 【修改】获取所有RRF融合的候选文档（不限制数量）
                List<Document> candidateDocs = documentRepository.findByIds(vectorResults);
                
                // 从 ScoredDocument 复制分数
                if (scoredDocs != null) {
                    copyScoresToDocuments(candidateDocs, scoredDocs);
                }
                
                // 根据关键词调整分数并排序
                if (!coreKeywords.isEmpty()) {
                    candidateDocs = adjustScoreByKeywords(candidateDocs, coreKeywords);
                    
                    if (candidateDocs.isEmpty()) {
                        return "抱歉，我在知识库中没有找到包含关键信息（" + String.join(", ", coreKeywords) + "）的内容。";
                    }
                    
                    // 【新增】过滤掉分数为0的文档（完全不匹配关键词的文档）
                    candidateDocs = candidateDocs.stream()
                        .filter(doc -> doc.getSimilarityScore() != null && doc.getSimilarityScore() > 0)
                        .collect(Collectors.toList());
                    
                    if (candidateDocs.isEmpty()) {
                        return "抱歉，我在知识库中没有找到包含关键信息（" + String.join(", ", coreKeywords) + "）的内容。";
                    }
                }
                
                // 【新增】按分数排序并截取Top-K
                candidateDocs.sort((a, b) -> Double.compare(
                    b.getSimilarityScore() != null ? b.getSimilarityScore() : 0,
                    a.getSimilarityScore() != null ? a.getSimilarityScore() : 0
                ));
                finalDocs = candidateDocs.stream()
                    .limit(topK)
                    .collect(Collectors.toList());
            }
                
            if (finalDocs.isEmpty()) {
                return "抱歉，我在知识库中没有找到相关的信息来回答您的问题。";
            }
            
            // 打印最终用于生成答案的文档及其分数
            log.info("=== Final Documents for Answer Generation ===");
            for (int i = 0; i < finalDocs.size(); i++) {
                Document doc = finalDocs.get(i);
                String contentPreview = doc.getParentContent() != null ? 
                    doc.getParentContent() : doc.getContent();
                log.info("Doc {}: id={}, filename={}, chunk={}/{}, preview={}, similarityScore={}",
                    i+1, doc.getId(), doc.getFilename(), 
                    doc.getChunkIndex()+1, doc.getTotalChunks(),
                    contentPreview.substring(0, Math.min(150, contentPreview.length())), doc.getSimilarityScore());
            }
            log.info("================================================");
                
            // 5. 构建上下文
            String context = buildContext(finalDocs);
            
            log.info("Retrieved {} documents for answer generation", finalDocs.size());
                
            // 6. 调用LLM生成答案
            String answer = chatClient.prompt()
                .system("""
                    你是一个智能助手，必须严格基于提供的参考资料回答用户问题。
                        
                    重要规则：
                    1. 只能使用参考资料中的信息，不得编造或添加资料中没有的内容
                    2. 如果参考资料中没有相关信息，明确告知用户“知识库中没有找到相关信息”
                    3. 回答时要引用参考资料的具体内容，保持准确性
                    4. 不要使用你自己的训练知识，只使用提供的参考资料
                    """)
                .user("""
                    问题：%s
                        
                    参考资料：
                    %s
                        
                    请基于以上参考资料回答问题：
                    """.formatted(question, context))
                .call()
                .content();
            
            log.info("Answer generated successfully");
            return answer;
                
        } catch (Exception e) {
            log.error("Failed to answer question", e);
            throw new RuntimeException("Question answering failed: " + e.getMessage(), e);
        }
    }
        
    /**
     * 根据内容查找文档
     */
    private Document findDocByContent(List<Document> docs, String content) {
        return docs.stream()
            .filter(doc -> doc.getContent().equals(content))
            .findFirst()
            .orElse(null);
    }
    
    /**
     * 删除文档
     */
    @Transactional
    public void deleteDocument(String filename) {
        try {
            log.info("Deleting document: {}", filename);
            
            // 获取要删除的文档ID
            List<Document> docs = documentRepository.findByFilename(filename);
            Set<Long> docIds = docs.stream()
                .map(Document::getId)
                .collect(Collectors.toSet());
            
            // 从数据库删除
            documentRepository.deleteByFilename(filename);
            
            // 从JVector索引删除（标记删除）
            jVectorService.removeVectorsForFilename(filename, docIds);
            
            log.info("Successfully deleted document: {} ({} chunks)", filename, docs.size());
            
        } catch (Exception e) {
            log.error("Failed to delete document: {}", filename, e);
            throw new RuntimeException("Document deletion failed: " + e.getMessage(), e);
        }
    }
    
    /**
     * 获取所有文档列表
     */
    public List<String> getAllDocuments() {
        return documentRepository.getAllFilenames();
    }
    
    /**
     * 获取文档详情
     */
    public List<Document> getDocumentChunks(String filename) {
        return documentRepository.findByFilename(filename);
    }
    
    /**
     * 从 ScoredDocument 列表中复制分数到 Document 对象
     */
    private void copyScoresToDocuments(List<Document> docs, List<com.myspringairag.model.ScoredDocument> scoredDocs) {
        Map<Long, Double> scoreMap = scoredDocs.stream()
            .collect(Collectors.toMap(
                com.myspringairag.model.ScoredDocument::getId,
                com.myspringairag.model.ScoredDocument::getFinalScore
            ));
        
        for (Document doc : docs) {
            doc.setSimilarityScore(scoreMap.getOrDefault(doc.getId(), 1.0));
        }
    }
    
    /**
     * 根据关键词调整相似度分数并排序
     * 公式：新分数 = 原分数 * (匹配关键词数 / 总关键词数)
     * @param candidates 候选文档列表
     * @param keywords 核心关键词列表
     * @return 调整分数后的文档列表（已按新分数降序排序）
     */
    private List<Document> adjustScoreByKeywords(List<Document> candidates, List<String> keywords) {
        if (keywords.isEmpty() || candidates.isEmpty()) {
            return candidates;
        }
        log.info("Keywords for score adjustment: {}", keywords);
        int totalKeywords = keywords.size();
        
        // 为每个文档计算新的相似度分数
        for (Document doc : candidates) {
            String content = (doc.getParentContent() != null ? doc.getParentContent() : doc.getContent()).toLowerCase();
            
            // 统计匹配的关键词数量
            int matchedCount = 0;
            for (String keyword : keywords) {
                if (content.contains(keyword.toLowerCase())) {
                    matchedCount++;
                }
            }
            
            // 获取原始分数（如果没有则使用默认值0)
            double originalScore = doc.getSimilarityScore() != null ? doc.getSimilarityScore() : 0;
            
            // 计算新分数：originalScore * (matchedCount / totalKeywords)
            double newScore = originalScore * ((double) matchedCount / totalKeywords);
            log.info("Adjusted score for doc {}: {}， originalScore: {}", doc.getId(), newScore, originalScore);
            doc.setSimilarityScore(newScore);
        }
        
        // 按新分数降序排序
        candidates.sort((a, b) -> Double.compare(
            b.getSimilarityScore() != null ? b.getSimilarityScore() : 0,
            a.getSimilarityScore() != null ? a.getSimilarityScore() : 0
        ));
        
        return candidates;
    }
    
    /**
     * 构建检索上下文（使用parentContent提供完整上下文）
     */
    private String buildContext(List<Document> docs) {
        StringBuilder context = new StringBuilder();
        
        // 使用LinkedHashSet去重，避免同一大块被多次返回
        Set<String> uniqueContexts = new LinkedHashSet<>();
        
        for (Document doc : docs) {
            // 优先使用parentContent，如果不存在则使用content
            String contextText = doc.getParentContent() != null ? 
                doc.getParentContent() : doc.getContent();
            uniqueContexts.add(contextText);
        }
        
        // 构建最终上下文
        int index = 1;
        for (String contextText : uniqueContexts) {
            context.append(String.format("[%d] %s\n\n", index++, contextText));
        }
        
        return context.toString();
    }
}
