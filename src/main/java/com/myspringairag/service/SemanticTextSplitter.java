package com.myspringairag.service;

import com.myspringairag.model.TextChunk;
import lombok.extern.slf4j.Slf4j;
import org.springframework.ai.embedding.EmbeddingModel;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.text.BreakIterator;
import java.util.*;

@Slf4j
@Service
public class SemanticTextSplitter {

    @Autowired
    private EmbeddingModel embeddingModel;
    
    private static final int DEFAULT_MAX_CHUNK_TOKENS = 512;
    private static final int DEFAULT_OVERLAP_TOKENS = 100;
    private static final float DEFAULT_SIMILARITY_THRESHOLD = 0.7f;
    private static final int MAX_SENTENCES_PER_CHUNK = 20;

    /**
     * 固定大小分块（基于句子边界）
     */
    public List<TextChunk> splitBySentences(String docId, String text, 
                                            int maxChunkTokens, int overlapTokens) {
        List<String> sentences = splitIntoSentences(text);
        List<TextChunk> chunks = new ArrayList<>();
        StringBuilder currentChunk = new StringBuilder();
        int currentTokens = 0;
        int chunkIndex = 0;

        for (String sentence : sentences) {
            int sentenceTokens = estimateTokenCount(sentence);

            if (currentTokens + sentenceTokens > maxChunkTokens && currentTokens > 0) {
                String chunkText = currentChunk.toString().trim();
                String chunkId = docId + "_" + chunkIndex++;
                chunks.add(new TextChunk(chunkId, docId, chunkIndex, chunkText, currentTokens));

                // 重叠处理：保留最后 overlapTokens 字符（近似）
                int overlapChars = Math.min(overlapTokens * 4, chunkText.length());
                String overlapText = chunkText.substring(Math.max(0, chunkText.length() - overlapChars));
                currentChunk = new StringBuilder(overlapText);
                currentTokens = estimateTokenCount(overlapText);
            }

            currentChunk.append(sentence).append(" ");
            currentTokens += sentenceTokens;
        }

        if (currentChunk.length() > 0) {
            String chunkText = currentChunk.toString().trim();
            String chunkId = docId + "_" + chunkIndex;
            chunks.add(new TextChunk(chunkId, docId, chunkIndex, chunkText, currentTokens));
        }

        log.info("Fixed-size splitting: {} sentences -> {} chunks", sentences.size(), chunks.size());
        return chunks;
    }

    /**
     * 语义分块（基于句子 embedding 相似度）
     */
    public List<TextChunk> semanticChunk(String docId, String text, 
                                         float similarityThreshold,
                                         int maxChunkTokens,
                                         int maxSentencesPerChunk) {
        List<String> sentences = splitIntoSentences(text);
        if (sentences.isEmpty()) {
            log.warn("No sentences found in text");
            return Collections.emptyList();
        }
        
        log.info("Semantic chunking: {} sentences to process", sentences.size());
        
        // 批量生成 embeddings（性能优化）
        List<float[]> embeddings = batchEmbed(sentences);
        
        List<TextChunk> chunks = new ArrayList<>();
        List<String> currentGroup = new ArrayList<>();
        float[] currentCentroid = null;
        int currentTokens = 0;
        int chunkIndex = 0;
        
        for (int i = 0; i < sentences.size(); i++) {
            String sentence = sentences.get(i);
            float[] emb = embeddings.get(i);
            int sentenceTokens = estimateTokenCount(sentence);
            
            if (currentGroup.isEmpty()) {
                currentGroup.add(sentence);
                currentCentroid = emb.clone();
                currentTokens = sentenceTokens;
            } else {
                float sim = cosineSimilarity(emb, currentCentroid);
                boolean forceBreak = false;
                
                // 语义相似度低于阈值 或 超过最大句子数 或 超过最大 token 数
                if (sim < similarityThreshold) {
                    forceBreak = true;
                }
                if (currentGroup.size() >= maxSentencesPerChunk) {
                    forceBreak = true;
                }
                if (currentTokens + sentenceTokens > maxChunkTokens) {
                    forceBreak = true;
                }
                
                if (forceBreak) {
                    // 保存当前块
                    String chunkText = String.join(" ", currentGroup);
                    String chunkId = docId + "_" + chunkIndex++;
                    chunks.add(new TextChunk(chunkId, docId, chunkIndex, chunkText, currentTokens));
                    
                    // 重置新块
                    currentGroup.clear();
                    currentGroup.add(sentence);
                    currentCentroid = emb.clone();
                    currentTokens = sentenceTokens;
                } else {
                    currentGroup.add(sentence);
                    currentTokens += sentenceTokens;
                    // 更新质心
                    currentCentroid = updateCentroid(currentCentroid, emb, currentGroup.size());
                }
            }
        }
        
        // 添加最后一个块
        if (!currentGroup.isEmpty()) {
            String chunkText = String.join(" ", currentGroup);
            String chunkId = docId + "_" + chunkIndex;
            chunks.add(new TextChunk(chunkId, docId, chunkIndex, chunkText, currentTokens));
        }
        
        log.info("Semantic chunking completed: {} sentences -> {} chunks", sentences.size(), chunks.size());
        return chunks;
    }

    /**
     * 批量生成 embedding 向量
     */
    private List<float[]> batchEmbed(List<String> texts) {
        List<float[]> embeddings = new ArrayList<>();
        
        // Spring AI 的 EmbeddingModel 通常一次只能处理一个文本
        // 如果需要批量，可以分批处理
        int batchSize = 10; // 每批处理10个文本
        for (int i = 0; i < texts.size(); i += batchSize) {
            int end = Math.min(i + batchSize, texts.size());
            List<String> batch = texts.subList(i, end);
            
            for (String text : batch) {
                try {
                    float[] embedding = embeddingModel.embed(text);
                    embeddings.add(embedding);
                } catch (Exception e) {
                    log.error("Failed to embed text: {}", text.substring(0, Math.min(50, text.length())), e);
                    // 使用零向量作为fallback
                    embeddings.add(new float[384]); // BGE-Small-ZH 维度
                }
            }
        }
        
        return embeddings;
    }

    /**
     * 使用 BreakIterator 进行中文分句（比正则更准确）
     */
    private List<String> splitIntoSentences(String text) {
        List<String> sentences = new ArrayList<>();
        BreakIterator iterator = BreakIterator.getSentenceInstance(Locale.CHINESE);
        iterator.setText(text);
        int start = iterator.first();
        for (int end = iterator.next(); end != BreakIterator.DONE; start = end, end = iterator.next()) {
            String sentence = text.substring(start, end).trim();
            if (!sentence.isEmpty()) {
                sentences.add(sentence);
            }
        }
        return sentences;
    }

    /**
     * 简化 token 估算（中文:1字符0.5token；英文:1单词1.3token）
     */
    private int estimateTokenCount(String text) {
        int chinese = (int) text.chars().filter(c -> c >= 0x4E00 && c <= 0x9FA5).count();
        int other = text.length() - chinese;
        // 中文按 0.5，英文+数字按 0.25（粗略）
        return (int)(chinese * 0.5 + other * 0.25);
    }

    private float cosineSimilarity(float[] a, float[] b) {
        if (a.length != b.length) {
            throw new IllegalArgumentException("Vector dimensions must match");
        }
        
        float dot = 0, normA = 0, normB = 0;
        for (int i = 0; i < a.length; i++) {
            dot += a[i] * b[i];
            normA += a[i] * a[i];
            normB += b[i] * b[i];
        }
        
        float denominator = (float) (Math.sqrt(normA) * Math.sqrt(normB));
        if (denominator < 1e-8) {
            return 0.0f;
        }
        
        return dot / denominator;
    }

    private float[] updateCentroid(float[] centroid, float[] newVec, int newSize) {
        float[] updated = new float[centroid.length];
        for (int i = 0; i < centroid.length; i++) {
            updated[i] = (centroid[i] * (newSize - 1) + newVec[i]) / newSize;
        }
        return updated;
    }
}
