package com.myspringairag.service;

import jakarta.annotation.PostConstruct;
import lombok.extern.slf4j.Slf4j;
import org.springframework.ai.embedding.EmbeddingModel;
import org.springframework.stereotype.Service;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

@Slf4j
@Service
public class EmbeddingService {
    
    private final EmbeddingModel embeddingModel;
    
    // LRU缓存：最多缓存100个embedding，自动淘汰最久未使用的
    private final Map<String, float[]> embeddingCache = new LinkedHashMap<>(100, 0.75f, true) {
        @Override
        protected boolean removeEldestEntry(Map.Entry<String, float[]> eldest) {
            return size() > 100; // 最多缓存100个embedding
        }
    };
    
    public EmbeddingService(EmbeddingModel embeddingModel) {
        this.embeddingModel = embeddingModel;
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
        synchronized (embeddingCache) {
            if (embeddingCache.containsKey(text)) {
                log.debug("Cache hit for text: {}", text.substring(0, Math.min(30, text.length())));
                return embeddingCache.get(text).clone(); // 返回副本，避免外部修改
            }
        }
        
        try {
            // 使用 Spring AI 的 EmbeddingModel
            List<float[]> embeddings = embeddingModel.embed(List.of(text));
            
            if (embeddings.isEmpty()) {
                throw new RuntimeException("Failed to generate embedding");
            }
            
            float[] vector = embeddings.get(0);
            
            // 存入缓存
            synchronized (embeddingCache) {
                embeddingCache.put(text, vector.clone());
            }
            
            log.debug("Generated embedding with dimension: {}", vector.length);
            return vector;
        } catch (Exception e) {
            log.error("Failed to generate embedding for text: {}", text.substring(0, Math.min(50, text.length())), e);
            throw new RuntimeException("Embedding generation failed", e);
        }
    }
    
    /**
     * 批量生成向量
     */
    public List<float[]> embedBatch(List<String> texts) {
        if (texts == null || texts.isEmpty()) {
            return List.of();
        }
        
        try {
            List<float[]> embeddings = embeddingModel.embed(texts);
            log.debug("Generated {} embeddings", embeddings.size());
            return embeddings;
        } catch (Exception e) {
            log.error("Failed to generate batch embeddings", e);
            throw new RuntimeException("Batch embedding generation failed", e);
        }
    }
}
