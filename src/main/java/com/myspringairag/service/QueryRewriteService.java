package com.myspringairag.service;

import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;

/**
 * 查询重写服务 - 将用户自然语言问题转换为更适合向量检索的格式
 */
@Slf4j
@Service
public class QueryRewriteService {
    
    /**
     * 重写用户查询，生成更适合向量检索的格式
     * @param userQuery 原始用户问题
     * @return 重写后的查询字符串
     */
    public String rewrite(String userQuery) {
        if (userQuery == null || userQuery.trim().isEmpty()) {
            return userQuery;
        }
        
        // 1. 清理查询（去除疑问词、标点等）
        String cleaned = cleanQuery(userQuery);
        
        // 2. 重构为陈述句格式
        String restructured = restructureQuestion(cleaned);
        
        // 3. 如果变化不大，返回原始查询（避免过度处理）
        if (restructured.equals(cleaned) || restructured.length() < cleaned.length() / 2) {
            return cleaned;
        }
        
        log.info("Query rewritten: '{}' -> '{}'", userQuery, restructured);
        return restructured;
    }
    
    /**
     * 清理查询：去除疑问词、语气词、标点符号
     */
    private String cleanQuery(String query) {
        return query
            // 去除开头疑问词
            .replaceAll("^(如何在|怎么|怎样|如何|请问|帮我|教我|告诉我|我想了解|我想知道)\\s*", "")
            // 去除标点符号
            .replaceAll("[?？。！，,、；:：\"\"''（）()]", "")
            // 规范化空格
            .replaceAll("\\s+", " ")
            .trim();
    }
    
    /**
     * 重构问题：将疑问句转为陈述句格式
     */
    private String restructureQuestion(String query) {
        // 模式1: "XXX如何YYY" → "XXX YYY 的方法"
        if (query.contains("如何") || query.contains("怎么") || query.contains("怎样")) {
            String result = query
                .replaceAll("(如何|怎么|怎样)", "")
                .trim();
            
            // 如果结果不以"方法"、"步骤"等结尾，添加"的方法"
            if (!result.matches(".*(?:方法|步骤|教程|指南|配置|安装|部署|方式)$")) {
                result = result + "的方法";
            }
            
            return result;
        }
        
        // 模式2: "XXX是什么" / "XXX的方式是什么" → 提取核心内容
        if (query.contains("是什么") || query.contains("是什么意思")) {
            String result = query
                .replaceAll("(是什么|是什么意思)[？?]?$", "")
                .trim();
            
            // 如果包含"方式"、"方法"、"步骤"等词，保留并优化
            if (result.matches(".*(?:方式|方法|步骤|教程|指南).*$")) {
                // 例如："启动open-webui的两种方式" → 保持不变或简化
                return result;
            } else {
                // 其他情况：添加"说明"或"介绍"
                return result + "的说明";
            }
        }
        
        // 模式3: 直接返回清理后的查询
        return query;
    }
}
