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

import java.io.IOException;
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
    
    /**
     * 存储当前任务的 taskId
     * 
     * 重要说明：
     * - 此 ThreadLocal 仅在 documentProcessingExecutor 单线程中使用
     * - 不应用于并行计算线程（parallelComputeExecutor）
     * - 如果需要跨线程传递 taskId，请显式传递参数而非依赖 ThreadLocal
     * - 原因：InheritableThreadLocal 在线程池复用场景下会导致数据污染
     * 
     * 使用场景：
     * 1. AsyncUploadService.processDocument() 设置 taskId
     * 2. RagService.uploadDocumentFromPath() 读取 taskId 用于进度推送
     * 3. RagService.processAndIndex() 读取 taskId 用于进度推送
     * 
     * @see AsyncUploadService#processDocument(String, Path, String)
     */
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
        int emptyChunkCount = 0;
        
        // 第1步：保存所有文档到数据库（过滤空文本）
        for (int i = 0; i < chunks.size(); i++) {
            String chunk = chunks.get(i);
            
            // 过滤空文本或纯空白文本
            if (chunk == null || chunk.trim().isEmpty()) {
                log.warn("Skipping empty chunk at index {} for file {}", i, filename);
                emptyChunkCount++;
                continue;
            }
            
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
        
        if (emptyChunkCount > 0) {
            log.warn("Filtered out {} empty chunks from file {}", emptyChunkCount, filename);
        }
        
        if (chunksToEmbed.isEmpty()) {
            throw new IllegalArgumentException("No valid chunks to embed after filtering empty texts");
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
     * 检索结果封装类
     */
    private static class RetrievalResult {
        List<Long> vectorResults;
        List<com.myspringairag.model.ScoredDocument> scoredDocs;
        List<Document> finalDocs;
            
        RetrievalResult(List<Long> vectorResults, 
                       List<com.myspringairag.model.ScoredDocument> scoredDocs,
                       List<Document> finalDocs) {
            this.vectorResults = vectorResults;
            this.scoredDocs = scoredDocs;
            this.finalDocs = finalDocs;
        }
    }
        
    /**
     * 基于知识库回答问题（查询转换 + 向量检索 + BGE重排序）
     */
    public String answerQuestion(String question) {
        try {
            log.info("Answering question: {}", question);
                
            // 执行检索和文档处理
            RetrievalResult result = retrieveAndProcessDocuments(question);
                
            // 检查是否有检索结果
            if (result.finalDocs.isEmpty()) {
                return "抱歉，我在知识库中没有找到相关的信息来回答您的问题。";
            }
                
            // 打印最终用于生成答案的文档及其分数
            logFinalDocuments(result.finalDocs);
                
            // 构建上下文
            String context = buildContext(result.finalDocs);
                
            log.info("Retrieved {} documents for answer generation", result.finalDocs.size());
                    
            // 调用LLM生成答案
            String answer = chatClient.prompt()
                .system("""
                    你是一个智能助手，必须严格基于提供的参考资料回答用户问题。
                            
                    重要规则：
                    1. 只能使用参考资料中的信息，不得编造或添加资料中没有的内容
                    2. 如果参考资料中没有相关信息，明确告知用户"知识库中没有找到相关信息"
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
     * 流式问答（SSE）
     */
    public void answerQuestionStream(String question, org.springframework.web.servlet.mvc.method.annotation.SseEmitter emitter) {
        try {
            log.info("Answering question (stream): {}", question);
                
            // 发送开始事件
            emitter.send(org.springframework.web.servlet.mvc.method.annotation.SseEmitter.event()
                .name("start")
                .data(Map.of("message", "开始生成答案...")));
                
            // 执行检索和文档处理
            RetrievalResult result = retrieveAndProcessDocuments(question);
                
            // 检查是否有检索结果
            if (result.finalDocs.isEmpty()) {
                emitter.send(org.springframework.web.servlet.mvc.method.annotation.SseEmitter.event()
                    .name("complete")
                    .data(Map.of("answer", "抱歉，我在知识库中没有找到相关的信息来回答您的问题。")));
                emitter.complete();
                return;
            }
                
            // 打印最终用于生成答案的文档及其分数
            logFinalDocuments(result.finalDocs);
                
            // 构建上下文
            String context = buildContext(result.finalDocs);
                
            // 流式调用LLM
            StringBuilder fullAnswer = new StringBuilder();
                
            chatClient.prompt()
                .system("""
                    你是一个智能助手，必须严格基于提供的参考资料回答用户问题。
                            
                    重要规则：
                    1. 只能使用参考资料中的信息，不得编造或添加资料中没有的内容
                    2. 如果参考资料中没有相关信息，明确告知用户"知识库中没有找到相关信息"
                    3. 回答时要引用参考资料的具体内容，保持准确性
                    4. 不要使用你自己的训练知识，只使用提供的参考资料
                    """)
                .user("""
                    问题：%s
                            
                    参考资料：
                    %s
                            
                    请基于以上参考资料回答问题：
                    """.formatted(question, context))
                .stream()
                .content()
                .doOnNext(chunk -> {
                    if (chunk != null && !chunk.isEmpty()) {
                        fullAnswer.append(chunk);
                        try {
                            emitter.send(org.springframework.web.servlet.mvc.method.annotation.SseEmitter.event()
                                .name("token")
                                .data(Map.of("token", chunk)));
                        } catch (IOException e) {
                            log.error("Failed to send token", e);
                        }
                    }
                })
                .doOnError(error -> {
                    log.error("Stream error", error);
                    try {
                        emitter.send(org.springframework.web.servlet.mvc.method.annotation.SseEmitter.event()
                            .name("error")
                            .data(Map.of("message", "生成答案时出错: " + error.getMessage())));
                    } catch (IOException e) {
                        log.error("Failed to send error event", e);
                    }
                    emitter.completeWithError(error);
                })
                .doOnComplete(() -> {
                    try {
                        emitter.send(org.springframework.web.servlet.mvc.method.annotation.SseEmitter.event()
                            .name("complete")
                            .data(Map.of("answer", fullAnswer.toString())));
                        emitter.complete();
                        log.info("Stream answer completed");
                    } catch (IOException e) {
                        log.error("Failed to send complete event", e);
                        emitter.completeWithError(e);
                    }
                })
                .subscribe();
                
        } catch (Exception e) {
            log.error("Failed to answer question (stream)", e);
            try {
                emitter.send(org.springframework.web.servlet.mvc.method.annotation.SseEmitter.event()
                    .name("error")
                    .data(Map.of("message", "回答失败: " + e.getMessage())));
            } catch (IOException ioException) {
                log.error("Failed to send error event", ioException);
            }
            emitter.completeWithError(e);
        }
    }
        
    /**
     * 执行文档检索和处理（公共逻辑）
     * @param question 用户问题
     * @return 检索结果（包含向量结果、评分文档和最终文档列表）
     */
    private RetrievalResult retrieveAndProcessDocuments(String question) {
        // 0. 查询重写
        String rewrittenQuery = queryRewriteService.rewrite(question);
        
        // 1. 向量检索
        VectorSearchResult searchResult = performVectorSearch(question, rewrittenQuery);
        
        // 如果没有检索结果，返回空结果
        if (searchResult.vectorResults.isEmpty()) {
            return new RetrievalResult(searchResult.vectorResults, searchResult.scoredDocs, List.of());
        }
        
        // 2. 文档处理和重排序
        List<Document> finalDocs = processAndRankDocuments(question, searchResult);
        
        return new RetrievalResult(searchResult.vectorResults, searchResult.scoredDocs, finalDocs);
    }
    
    /**
     * 向量搜索结果封装
     */
    private static class VectorSearchResult {
        List<Long> vectorResults;
        List<com.myspringairag.model.ScoredDocument> scoredDocs;
        
        VectorSearchResult(List<Long> vectorResults, List<com.myspringairag.model.ScoredDocument> scoredDocs) {
            this.vectorResults = vectorResults;
            this.scoredDocs = scoredDocs;
        }
    }
    
    /**
     * 执行向量检索
     */
    private VectorSearchResult performVectorSearch(String question, String rewrittenQuery) {
        long vectorSearchStart = System.currentTimeMillis();
        int searchTopK = useReranking ? rerankTopK : topK;
        
        List<Long> vectorResults;
        List<com.myspringairag.model.ScoredDocument> scoredDocs = null;
        
        if (useParallelRetrieval && parallelRetrievalService != null) {
            // 并行多查询变体检索
            scoredDocs = parallelRetrievalService.parallelSearch(question, rewrittenQuery, searchTopK);
            logVectorSearchResults(scoredDocs);
            
            vectorResults = scoredDocs.stream()
                .map(com.myspringairag.model.ScoredDocument::getId)
                .collect(Collectors.toList());
            
            log.info("Parallel retrieval returned {} results (took: {}ms)", 
                vectorResults.size(), System.currentTimeMillis() - vectorSearchStart);
        } else {
            // 传统单查询检索
            float[] queryVector = embeddingService.embed(rewrittenQuery);
            Map<Long, Float> docScores = jVectorService.searchWithScores(queryVector, searchTopK);
            
            scoredDocs = convertToScoredDocuments(docScores);
            vectorResults = scoredDocs.stream()
                .map(com.myspringairag.model.ScoredDocument::getId)
                .collect(Collectors.toList());
        }
        
        return new VectorSearchResult(vectorResults, scoredDocs);
    }
    
    /**
     * 记录向量检索结果日志
     */
    private void logVectorSearchResults(List<com.myspringairag.model.ScoredDocument> scoredDocs) {
        log.info("Using parallel multi-query retrieval");
        log.info("=== Vector Search Results with Scores ===");
        for (int i = 0; i < scoredDocs.size(); i++) {
            com.myspringairag.model.ScoredDocument sd = scoredDocs.get(i);
            log.info("Rank {}: docId={}, originalScore={}, rrfScore={}, finalScore={}",
                i+1, sd.getId(), sd.getOriginalScore(), sd.getRrfScore(), sd.getFinalScore());
        }
        log.info("===========================================");
    }
    
    /**
     * 将Map转换为ScoredDocument列表
     */
    private List<com.myspringairag.model.ScoredDocument> convertToScoredDocuments(Map<Long, Float> docScores) {
        return docScores.entrySet().stream()
            .map(entry -> new com.myspringairag.model.ScoredDocument(
                entry.getKey(),
                null,
                entry.getValue(),
                entry.getValue(),
                entry.getValue()
            ))
            .collect(Collectors.toList());
    }
    
    /**
     * 处理候选文档并排序（包含重排序和关键词调整）
     */
    private List<Document> processAndRankDocuments(String question, VectorSearchResult searchResult) {
        List<String> coreKeywords = tokenizationService.extractCoreKeywords(
            queryRewriteService.rewrite(question));
        
        List<Document> candidateDocs = documentRepository.findByIds(searchResult.vectorResults);
        
        // 复制分数
        if (searchResult.scoredDocs != null) {
            copyScoresToDocuments(candidateDocs, searchResult.scoredDocs);
        }
        
        // 关键词分数调整
        candidateDocs = adjustCandidateScores(candidateDocs, coreKeywords);
        if (candidateDocs.isEmpty()) {
            return List.of();
        }
        
        // 重排序或直接排序
        if (useReranking && reranker != null) {
            return rerankDocuments(question, candidateDocs);
        } else {
            return sortAndLimitCandidates(candidateDocs);
        }
    }
    
    /**
     * 调整候选文档分数（关键词过滤）
     */
    private List<Document> adjustCandidateScores(List<Document> candidateDocs, List<String> coreKeywords) {
        if (!coreKeywords.isEmpty()) {
            candidateDocs = adjustScoreByKeywords(candidateDocs, coreKeywords);
            
            if (candidateDocs.isEmpty()) {
                return List.of();
            }
            
            // 过滤掉分数为0的文档
            candidateDocs = candidateDocs.stream()
                .filter(doc -> doc.getSimilarityScore() != null && doc.getSimilarityScore() > 0)
                .collect(Collectors.toList());
        }
        return candidateDocs;
    }
    
    /**
     * 使用重排序模型对文档进行重排序
     */
    private List<Document> rerankDocuments(String question, List<Document> candidateDocs) {
        try {
            log.info("Using reranker to refine {} candidates", candidateDocs.size());
            logCandidateDocuments(candidateDocs);
            
            // 构建重排序输入
            String[] passages = candidateDocs.stream()
                .map(doc -> doc.getParentContent() != null ? doc.getParentContent() : doc.getContent())
                .toArray(String[]::new);
            
            // 执行重排序
            Map<String, Float> rankedScores = reranker.rerankBatch(question, passages);
            
            // 取 top-K
            List<Document> finalDocs = rankedScores.entrySet().stream()
                .limit(topK)
                .map(entry -> findDocByContent(candidateDocs, entry.getKey()))
                .filter(Objects::nonNull)
                .collect(Collectors.toList());
            
            log.info("After reranking: {} documents selected", finalDocs.size());
            return finalDocs;
            
        } catch (Exception e) {
            log.error("Reranking failed, fallback to score-based sorting", e);
            // 重排序失败时，降级为按分数排序
            return sortAndLimitCandidates(candidateDocs);
        }
    }
    
    /**
     * 记录候选文档日志
     */
    private void logCandidateDocuments(List<Document> candidateDocs) {
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
    }
    
    /**
     * 按分数排序并限制返回数量
     */
    private List<Document> sortAndLimitCandidates(List<Document> candidateDocs) {
        candidateDocs.sort((a, b) -> Double.compare(
            b.getSimilarityScore() != null ? b.getSimilarityScore() : 0,
            a.getSimilarityScore() != null ? a.getSimilarityScore() : 0
        ));
        return candidateDocs.stream()
            .limit(topK)
            .collect(Collectors.toList());
    }
    
    /**
     * 打印最终用于生成答案的文档及其分数
     * @param finalDocs 最终文档列表
     */
    private void logFinalDocuments(List<Document> finalDocs) {
        if (finalDocs == null || finalDocs.isEmpty()) {
            return;
        }
        
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
            log.debug("Adjusted score for doc {}: {}， originalScore: {}", doc.getId(), newScore, originalScore);
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
