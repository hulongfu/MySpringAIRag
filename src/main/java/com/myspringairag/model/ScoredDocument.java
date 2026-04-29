package com.myspringairag.model;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

/**
 * 带分数的文档
 */
@Data
@AllArgsConstructor
@NoArgsConstructor
public class ScoredDocument {
    private Long id;
    private String content;
    private double originalScore;  // 原始向量相似度分数
    private double rrfScore;       // RRF融合分数
    private double finalScore;     // 最终综合分数
}
