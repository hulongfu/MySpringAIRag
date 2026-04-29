package com.myspringairag.service;

import com.myspringairag.model.TextChunk;
import lombok.extern.slf4j.Slf4j;
import org.springframework.ai.document.Document;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.util.*;
import java.util.stream.Collectors;

/**
 * 层级分块服务 - 实现Parent-Child Chunking策略
 * 
 * 核心思路：
 * 1. 先用语义分块生成大块（保持语义完整）
 * 2. 如果大块 > 200 tokens，按行拆分成小块
 * 3. 小块用于向量检索，大块用于提供上下文
 */
@Slf4j
@Service
public class HierarchicalChunkingService {
    
    @Autowired
    private SemanticTextSplitter semanticSplitter;
    
    // 配置参数
    private static final int SMALL_CHUNK_TOKENS = 200;   // 小块大小
    private static final int OVERLAP_TOKENS = 20;        // 重叠大小
    private static final int LONG_LINE_THRESHOLD = 300;  // 超长行阈值
    private static final double TOKEN_ESTIMATE_THRESHOLD = 0.85;  // 混合策略阈值
    
    /**
     * 层级分块
     * @param content 原始文档内容
     * @param filename 文件名
     * @return 分块后的Document列表（每个Document的content是小块，parentContent是大块）
     */
    public List<Document> hierarchicalChunk(String content, String filename) {
        log.info("Starting hierarchical chunking for: {}", filename);
        
        // 第一步：语义分块生成大块
        String docId = "doc_" + System.currentTimeMillis();
        List<TextChunk> largeChunks = semanticSplitter.semanticChunk(
            docId,
            content,
            0.7f,  // similarityThreshold
            800,   // maxChunkTokens (不限制大块大小)
            50     // maxSentencesPerChunk
        );
        log.info("Semantic splitting produced {} large chunks", largeChunks.size());
        
        // 第二步：对每个大块进行处理
        List<Document> result = new ArrayList<>();
        int globalChunkIndex = 0;
        
        for (TextChunk largeChunk : largeChunks) {
            String largeChunkText = largeChunk.getContent();
            int largeChunkTokens = estimateTokens(largeChunkText);
            
            if (largeChunkTokens <= SMALL_CHUNK_TOKENS) {
                // 小块直接作为结果，parentContent = content
                org.springframework.ai.document.Document doc = new org.springframework.ai.document.Document(
                    largeChunkText,
                    Map.of(
                        "filename", filename,
                        "chunkIndex", String.valueOf(globalChunkIndex),
                        "totalChunks", "0",  // 稍后更新
                        "content", largeChunkText,
                        "parentContent", largeChunkText
                    )
                );
                result.add(doc);
                globalChunkIndex++;
            } else {
                // 大块需要二次切分成小块
                List<String> smallChunks = splitLargeChunkToSmallChunks(largeChunkText);
                log.debug("Large chunk ({} tokens) split into {} small chunks", 
                    largeChunkTokens, smallChunks.size());
                
                for (String smallChunk : smallChunks) {
                    org.springframework.ai.document.Document doc = new org.springframework.ai.document.Document(
                        smallChunk,
                        Map.of(
                            "filename", filename,
                            "chunkIndex", String.valueOf(globalChunkIndex),
                            "totalChunks", "0",
                            "content", smallChunk,
                            "parentContent", largeChunkText
                        )
                    );
                    result.add(doc);
                    globalChunkIndex++;
                }
            }
        }
        
        // 更新totalChunks
        int totalChunks = result.size();
        for (Document doc : result) {
            doc.getMetadata().put("totalChunks", String.valueOf(totalChunks));
        }
        
        log.info("Hierarchical chunking completed: {} small chunks from {} large chunks", 
            result.size(), largeChunks.size());
        
        return result;
    }
    
    /**
     * 将大块拆分成小块（按行拆分，保持行完整性）
     */
    private List<String> splitLargeChunkToSmallChunks(String largeChunk) {
        // 按行拆分
        List<String> lines = splitByLines(largeChunk);
        
        // 合并行成小块（带重叠）
        return mergeLinesToChunks(lines, SMALL_CHUNK_TOKENS, OVERLAP_TOKENS);
    }
    
    /**
     * 按行拆分文本
     */
    private List<String> splitByLines(String text) {
        return Arrays.stream(text.split("\n"))
            .map(String::trim)
            .filter(line -> !line.isEmpty())
            .collect(Collectors.toList());
    }
    
    /**
     * 将行合并成小块，确保不切断行（除非行本身超长）
     * @param lines 行列表
     * @param maxTokens 最大token数
     * @param overlapTokens 重叠token数
     * @return 小块列表
     */
    private List<String> mergeLinesToChunks(List<String> lines, int maxTokens, int overlapTokens) {
        List<String> chunks = new ArrayList<>();
        StringBuilder currentChunk = new StringBuilder();
        int currentTokens = 0;
        
        for (String line : lines) {
            int lineTokens = estimateTokens(line);
            
            // 处理超长行
            if (lineTokens > LONG_LINE_THRESHOLD) {
                // 保存当前chunk
                if (currentChunk.length() > 0) {
                    chunks.add(currentChunk.toString().trim());
                    currentChunk = new StringBuilder();
                    currentTokens = 0;
                }
                // 超长行单独作为一个chunk（允许截断）
                chunks.add(line);
                continue;
            }
            
            // 如果加上这一行会超限，保存当前chunk并开始新的
            if (currentTokens + lineTokens > maxTokens && currentTokens > 0) {
                chunks.add(currentChunk.toString().trim());
                
                // 保留末尾部分作为重叠
                String overlap = getTailText(currentChunk.toString(), overlapTokens);
                currentChunk = new StringBuilder(overlap);
                currentTokens = estimateTokens(overlap);
            }
            
            // 添加到当前chunk
            if (currentChunk.length() > 0) {
                currentChunk.append("\n");
            }
            currentChunk.append(line);
            currentTokens += lineTokens;
        }
        
        // 添加最后一个chunk
        if (currentChunk.length() > 0) {
            chunks.add(currentChunk.toString().trim());
        }
        
        return chunks;
    }
    
    /**
     * 获取文本末尾部分（用于重叠）
     */
    private String getTailText(String text, int maxTokens) {
        String[] lines = text.split("\n");
        StringBuilder tail = new StringBuilder();
        int tokens = 0;
        
        // 从最后一行开始往前取
        for (int i = lines.length - 1; i >= 0 && tokens < maxTokens; i--) {
            String line = lines[i].trim();
            if (line.isEmpty()) continue;
            
            int lineTokens = estimateTokens(line);
            if (tokens + lineTokens > maxTokens && tokens > 0) {
                break;
            }
            
            if (tail.length() > 0) {
                tail.insert(0, "\n");
            }
            tail.insert(0, line);
            tokens += lineTokens;
        }
        
        return tail.toString().trim();
    }
    
    /**
     * 估算token数（混合策略）
     * 1. 先用简单估算快速判断
     * 2. 接近阈值时用tokenizer精确计算（TODO: 后续可以集成BGE tokenizer）
     */
    private int estimateTokens(String text) {
        if (text == null || text.isEmpty()) {
            return 0;
        }
        
        // 简单估算：中文≈0.7 token/字符，英文≈0.3 token/字符
        int chinese = (int) text.chars()
            .filter(c -> c >= 0x4E00 && c <= 0x9FA5)
            .count();
        int others = text.length() - chinese;
        int estimated = (int) (chinese * 0.7 + others * 0.3);
        
        // TODO: 如果需要更精确，可以在这里集成BGE tokenizer
        // if (estimated > SMALL_CHUNK_TOKENS * TOKEN_ESTIMATE_THRESHOLD) {
        //     return bgeTokenizer.encode(text).length;
        // }
        
        return estimated;
    }
}
