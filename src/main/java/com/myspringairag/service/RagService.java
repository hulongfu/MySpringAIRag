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
        
        // 3. 为每个chunk生成向量并存储
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
            
            // 生成向量（对小快content生成向量）
            float[] vector = embeddingService.embed(chunk);
            
            // 添加到JVector索引
            jVectorService.addVector(savedDoc.getId(), vector);
            
            log.debug("Processed chunk {}/{} for {}", i + 1, chunks.size(), filename);
        }
        
        // 90% - 向量化和存储完成
        if (taskId != null) {
            sseController.notifyProgress(taskId, 90, "正在构建向量索引...");
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
            long rewriteStart = System.currentTimeMillis();
            String rewrittenQuery = queryRewriteService.rewrite(question);
            if (!rewrittenQuery.equals(question)) {
                log.info("Query rewritten: '{}' -> '{}'", question, rewrittenQuery);
            }
            log.debug("Query rewrite took: {}ms", System.currentTimeMillis() - rewriteStart);
                
            // 1. 向量检索
            long vectorSearchStart = System.currentTimeMillis();
            int searchTopK = useReranking ? rerankTopK : topK;
            
            List<Long> vectorResults;
            
            // 如果启用并行检索，使用多查询变体
            if (useParallelRetrieval && parallelRetrievalService != null) {
                log.info("Using parallel multi-query retrieval");
                List<com.myspringairag.model.ScoredDocument> scoredDocs = 
                    parallelRetrievalService.parallelSearch(rewrittenQuery, searchTopK);
                
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
                log.debug("Embedding generation took: {}ms", System.currentTimeMillis() - vectorSearchStart);
                
                vectorResults = jVectorService.search(queryVector, searchTopK);
                log.info("Vector search returned {} results (took: {}ms)", 
                    vectorResults.size(), System.currentTimeMillis() - vectorSearchStart);
            }
                
            if (vectorResults.isEmpty()) {
                return "抱歉，我在知识库中没有找到相关的信息来回答您的问题。";
            }
            
            // 提取核心关键词
            List<String> coreKeywords = tokenizationService.extractCoreKeywords(rewrittenQuery);
            log.info("Core keywords for filtering: {}", coreKeywords);
                
            List<Document> finalDocs;
                
            // 2. 如果启用重排序，使用 BgeReranker 精排
            if (useReranking && reranker != null) {
                long rerankStart = System.currentTimeMillis();
                log.info("Using reranker to refine {} candidates", vectorResults.size());
                            
                // 获取候选文档内容
                List<Document> candidateDocs = documentRepository.findByIds(vectorResults);
                
                // 关键词过滤
                if (!coreKeywords.isEmpty()) {
                    candidateDocs = filterByKeywords(candidateDocs, coreKeywords);
                    log.info("After keyword filtering: {} documents", candidateDocs.size());
                    
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
                    log.info("Candidate {}: id={}, filename={}, chunk={}/{}, content_preview={}",
                        i+1, doc.getId(), doc.getFilename(), 
                        doc.getChunkIndex()+1, doc.getTotalChunks(),
                        contentPreview.substring(0, Math.min(100, contentPreview.length())));
                }
                log.info("===========================");
                            
                // 构建重排序输入（使用parentContent如果存在）
                String[] passages = candidateDocs.stream()
                    .map(doc -> doc.getParentContent() != null ? doc.getParentContent() : doc.getContent())
                    .toArray(String[]::new);
                            
                // 执行重排序
                Map<String, Float> rankedScores = reranker.rerankBatch(question, passages);
                            
                log.info("Reranking completed, scores: {}", rankedScores);
                            
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
                List<Document> candidateDocs = documentRepository.findByIds(vectorResults.subList(0, Math.min(topK, vectorResults.size())));
                
                // 关键词过滤
                if (!coreKeywords.isEmpty()) {
                    candidateDocs = filterByKeywords(candidateDocs, coreKeywords);
                    log.info("After keyword filtering: {} documents", candidateDocs.size());
                    
                    if (candidateDocs.isEmpty()) {
                        return "抱歉，我在知识库中没有找到包含关键信息（" + String.join(", ", coreKeywords) + "）的内容。";
                    }
                }
                
                finalDocs = candidateDocs;
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
                log.info("Doc {}: id={}, filename={}, chunk={}/{}, preview={}",
                    i+1, doc.getId(), doc.getFilename(), 
                    doc.getChunkIndex()+1, doc.getTotalChunks(),
                    contentPreview.substring(0, Math.min(150, contentPreview.length())));
            }
            log.info("================================================");
                
            // 5. 构建上下文
            String context = buildContext(finalDocs);
                
            // 调试：打印实际检索到的内容
            log.info("=== Retrieved Context ===");
            log.info(context);
            log.info("=========================");
                
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
                
            log.info("Generated answer with {} relevant documents", finalDocs.size());
            return answer;
                
        } catch (Exception e) {
            log.error("Failed to answer question", e);
            throw new RuntimeException("Question answering failed: " + e.getMessage(), e);
        }
    }
    
    /**
     * RRF (Reciprocal Rank Fusion) 算法融合多路检索结果
     * @param vectorResults 向量检索结果
     * @param keywordResults 关键词检索结果
     * @param topK 返回的最大结果数
     * @return 融合后的文档ID列表
     */
    private List<Long> reciprocalRankFusion(List<Long> vectorResults, List<Long> keywordResults, int topK) {
        // RRF参数
        final double K = 60.0;                    // RRF常数（经验值）
        final double VECTOR_WEIGHT = 0.7;         // 向量检索权重
        final double KEYWORD_WEIGHT = 0.3;        // 关键词检索权重
        
        // 计算每个文档的RRF分数
        Map<Long, Double> rrfScores = new HashMap<>();
        
        // 向量检索的RRF贡献（加权）
        for (int i = 0; i < vectorResults.size(); i++) {
            Long docId = vectorResults.get(i);
            double score = (1.0 / (K + i + 1)) * VECTOR_WEIGHT;
            rrfScores.merge(docId, score, Double::sum);
        }
        
        // 关键词检索的RRF贡献（加权）
        for (int i = 0; i < keywordResults.size(); i++) {
            Long docId = keywordResults.get(i);
            double score = (1.0 / (K + i + 1)) * KEYWORD_WEIGHT;
            rrfScores.merge(docId, score, Double::sum);
        }
        
        // 按RRF分数降序排序，取topK
        return rrfScores.entrySet().stream()
            .sorted(Map.Entry.<Long, Double>comparingByValue().reversed())
            .limit(topK)
            .map(Map.Entry::getKey)
            .collect(Collectors.toList());
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
     * 根据关键词过滤文档（分级匹配策略）
     * @param candidates 候选文档列表
     * @param keywords 核心关键词列表
     * @return 过滤后的文档列表
     */
    private List<Document> filterByKeywords(List<Document> candidates, List<String> keywords) {
        if (keywords.isEmpty() || candidates.isEmpty()) {
            return candidates;
        }
        
        int minMatchCount = Math.max(keywords.size() / 2, 1);  // 至少匹配一半的关键词
        log.debug("Keyword filtering: require at least {} matches from {} keywords", minMatchCount, keywords.size());
        
        // 按匹配的关键词数量分组（降序）
        Map<Integer, List<Document>> grouped = new TreeMap<>(Collections.reverseOrder());
        
        for (Document doc : candidates) {
            String content = (doc.getParentContent() != null ? doc.getParentContent() : doc.getContent()).toLowerCase();
            int matchCount = 0;
            
            for (String keyword : keywords) {
                if (content.contains(keyword.toLowerCase())) {
                    matchCount++;
                }
            }
            
            grouped.computeIfAbsent(matchCount, k -> new ArrayList<>()).add(doc);
        }
        
        // 从最高匹配数开始查找，返回第一个满足最低阈值的组
        for (int matchCount : grouped.keySet()) {
            if (matchCount >= minMatchCount) {
                log.info("Found {} documents with {} matching keywords (required: {})", 
                    grouped.get(matchCount).size(), matchCount, minMatchCount);
                return grouped.get(matchCount);
            }
        }
        
        // 如果没有找到足够匹配的文档，返回空列表
        log.warn("No documents found with at least {} matching keywords", minMatchCount);
        return Collections.emptyList();
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
