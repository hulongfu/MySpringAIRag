package com.myspringairag.model;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

/**
 * 文本分块模型
 */
@Data
@NoArgsConstructor
@AllArgsConstructor
public class TextChunk {
    private String chunkId;      // 分块ID
    private String docId;        // 文档ID
    private int chunkIndex;      // 分块索引
    private String content;      // 分块内容
    private int tokenCount;      // Token数量
}
