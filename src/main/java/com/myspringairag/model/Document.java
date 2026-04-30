package com.myspringairag.model;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.time.LocalDateTime;

@Data
@NoArgsConstructor
@AllArgsConstructor
public class Document {
    private Long id;
    private String filename;
    private String content;  // 小块（200 tokens，用于向量检索）
    private String parentContent;  // 大块（语义完整，用于提供上下文）
    private Integer chunkIndex;
    private Integer totalChunks;
    private LocalDateTime uploadTime;
    private Long fileSize;
    private String mimeType;
    private Double similarityScore;  // 相似度分数（用于排序）
}
