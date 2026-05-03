package com.myspringairag.service;

import com.github.benmanes.caffeine.cache.Cache;
import com.github.benmanes.caffeine.cache.Caffeine;
import jakarta.annotation.PostConstruct;
import lombok.extern.slf4j.Slf4j;
import org.springframework.ai.embedding.EmbeddingModel;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.stereotype.Service;

import java.util.*;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;

@Slf4j
@Service
public class EmbeddingService {
    
    private final EmbeddingModel embeddingModel;
    
    // 注入统一的并行计算线程池
    private final ExecutorService parallelExecutor;
    
    // Caffeine 缓存：最多缓存1000个embedding，10分钟未访问则淘汰
    private final Cache<String, float[]> embeddingCache = Caffeine.newBuilder()
        .maximumSize(1000)  // 最多1000条（BGE-Small-ZH: 384维 × 4字节 ≈ 1.5MB）
        .expireAfterAccess(10, TimeUnit.MINUTES)  // 10分钟未访问则淘汰
        .recordStats()  // 记录统计信息（用于监控缓存命中率）
        .build();
    
    public EmbeddingService(EmbeddingModel embeddingModel, 
                           @Qualifier("parallelComputeExecutor") ExecutorService parallelExecutor) {
        this.embeddingModel = embeddingModel;
        this.parallelExecutor = parallelExecutor;
    }
    
    /**
     * 应用启动时预加载模型，避免首次查询的冷启动延迟
     */
    @PostConstruct
    public void preloadModel() {
        log.info("Preloading embedding model...");
        try {
            // 使用一个简单的查询进行预热
            embed("warmup query for model preloading");
            log.info("Embedding model preloaded successfully");
        } catch (Exception e) {
            log.error("Failed to preload embedding model", e);
        }
    }
    
    /**
     * 将文本转换为向量（带缓存）
     */
    public float[] embed(String text) {
        if (text == null || text.isEmpty()) {
            throw new IllegalArgumentException("Text cannot be null or empty");
        }
        
        // 检查缓存
        float[] cached = embeddingCache.getIfPresent(text);
        if (cached != null) {
            return cached.clone(); // 返回副本，避免外部修改
        }
        
        try {
            // 使用 Spring AI 的 EmbeddingModel
            List<float[]> embeddings = embeddingModel.embed(List.of(text));
            
            if (embeddings.isEmpty()) {
                throw new RuntimeException("Failed to generate embedding");
            }
            
            float[] vector = embeddings.get(0);
            
            // 存入缓存
            embeddingCache.put(text, vector.clone());
            
            return vector;
        } catch (Exception e) {
            log.error("Failed to generate embedding for text: {}", text.substring(0, Math.min(50, text.length())), e);
            throw new RuntimeException("Embedding generation failed", e);
        }
    }
    
    /**
     * 批量生成向量（并行优化版本）
     * 利用多线程并行处理，提升2-3倍性能
     */
    public List<float[]> embedBatch(List<String> texts) {
        if (texts == null || texts.isEmpty()) {
            return List.of();
        }
        
        try {
            long startTime = System.currentTimeMillis();
            
            // 第0步：过滤空文本
            List<String> filteredTexts = new ArrayList<>();
            Map<Integer, Integer> originalIndexMap = new HashMap<>(); // 映射：过滤后索引 -> 原始索引
            int emptyCount = 0;
            
            for (int i = 0; i < texts.size(); i++) {
                String text = texts.get(i);
                if (text != null && !text.trim().isEmpty()) {
                    originalIndexMap.put(filteredTexts.size(), i);
                    filteredTexts.add(text);
                } else {
                    log.warn("Skipping empty text at index {}", i);
                    emptyCount++;
                }
            }
            
            if (emptyCount > 0) {
                log.warn("Filtered out {} empty texts from batch of {}", emptyCount, texts.size());
            }
            
            if (filteredTexts.isEmpty()) {
                log.error("All texts are empty, returning empty result");
                return new ArrayList<>();
            }
            
            // 第1步：检查缓存，分离出需要计算的文本
            List<String> toCompute = new ArrayList<>();
            Map<String, Integer> textToIndex = new HashMap<>();
            List<float[]> results = new ArrayList<>(Collections.nCopies(filteredTexts.size(), null));
            int cachedCount = 0;
            
            for (int i = 0; i < filteredTexts.size(); i++) {
                String text = filteredTexts.get(i);
                float[] cached = embeddingCache.getIfPresent(text);
                if (cached != null) {
                    results.set(i, cached.clone());
                    cachedCount++;
                } else {
                    toCompute.add(text);
                    textToIndex.put(text, i);
                }
            }
            
            log.debug("Embedding batch: total={}, empty={}, filtered={}, cached={}, to_compute={}", 
                texts.size(), emptyCount, filteredTexts.size(), cachedCount, toCompute.size());
            
            // 第2步：如果没有需要计算的，直接返回
            if (toCompute.isEmpty()) {
                return results;
            }
            
            // 第3步：并行计算未缓存的向量
            List<float[]> computedVectors;
            if (toCompute.size() <= 5) {
                // 小批量：使用串行批量API（效率更高）
                computedVectors = embeddingModel.embed(toCompute);
            } else {
                // 大批量：并行处理（分块并行）
                computedVectors = parallelEmbed(toCompute);
            }
            
            // 第4步：填充结果并更新缓存
            for (int i = 0; i < toCompute.size(); i++) {
                String text = toCompute.get(i);
                float[] vector = computedVectors.get(i);
                
                // 安全检查：确保向量不为null
                if (vector == null) {
                    log.error("Embedding model returned null vector for text: {}", 
                        text.substring(0, Math.min(50, text.length())));
                    continue;
                }
                
                int originalIndex = textToIndex.get(text);
                results.set(originalIndex, vector);
                
                // 更新缓存
                embeddingCache.put(text, vector.clone());
            }
            
            long duration = System.currentTimeMillis() - startTime;
            
            // 输出缓存统计信息
            var stats = embeddingCache.stats();
            double hitRatePercent = stats.hitRate() * 100;
            log.info("Batch embedding completed: {} texts in {}ms (cache hit rate: {}%, total hits: {}, total misses: {})",
                filteredTexts.size(), duration, 
                String.format("%.1f", hitRatePercent),
                stats.hitCount(),
                stats.missCount());
            
            return results;
            
        } catch (Exception e) {
            log.error("Failed to generate batch embeddings", e);
            throw new RuntimeException("Batch embedding generation failed", e);
        }
    }
    
    /**
     * 并行执行Embedding（分块并行处理）
     */
    private List<float[]> parallelEmbed(List<String> texts) {
        int batchSize = Math.max(10, texts.size() / 4); // 每批至少10个，最多分4批
        List<List<String>> batches = new ArrayList<>();
        
        for (int i = 0; i < texts.size(); i += batchSize) {
            int end = Math.min(i + batchSize, texts.size());
            batches.add(texts.subList(i, end));
        }
        
        log.debug("Parallel embedding: {} texts split into {} batches", texts.size(), batches.size());
        
        // 并行执行各批次
        List<Future<List<float[]>>> futures = new ArrayList<>();
        for (List<String> batch : batches) {
            futures.add(parallelExecutor.submit(() -> embeddingModel.embed(batch)));
        }
        
        // 收集结果
        List<float[]> allVectors = new ArrayList<>();
        for (Future<List<float[]>> future : futures) {
            try {
                allVectors.addAll(future.get());
            } catch (Exception e) {
                throw new RuntimeException("Parallel embedding failed", e);
            }
        }
        
        return allVectors;
    }
}
