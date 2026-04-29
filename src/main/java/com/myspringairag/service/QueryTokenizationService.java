package com.myspringairag.service;

import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;
import org.wltea.analyzer.core.IKSegmenter;
import org.wltea.analyzer.core.Lexeme;

import java.io.IOException;
import java.io.StringReader;
import java.util.*;
import java.util.stream.Collectors;

/**
 * 查询分词服务 - 使用IK分词器进行中文分词
 */
@Slf4j
@Service
public class QueryTokenizationService {
    
    // IK分词器（智能分词模式）
    private static final boolean USE_SMART_MODE = true;
    
    // 停用词集合
    private static final Set<String> STOP_WORDS = new HashSet<>(Arrays.asList(
        "的", "了", "是", "在", "我", "有", "和", "就", "不", "人", "都", "一", "一个",
        "上", "也", "很", "到", "说", "要", "去", "你", "会", "着", "没有", "看", "好",
        "自己", "这", "那", "它", "他", "她", "什么", "怎么", "为什么", "如何",
        "中", "与", "及", "等", "等等", "或", "或者", "但", "但是", "而",
        "如果", "因为", "所以", "虽然", "然后", "接着", "之后", "之前"
    ));
    
    // 自定义术语词典（技术相关）
    private static final Set<String> CUSTOM_TERMS = new HashSet<>(Arrays.asList(
        "华为云", "阿里云", "腾讯云", "百度云",
        "云桌面", "云计算", "云服务",
        "docker", "Docker", "kubernetes", "Kubernetes", "k8s",
        "ollama", "Ollama", "langchain", "LangChain",
        "spring", "Spring", "boot", "Boot", "ai", "AI",
        "mysql", "MySQL", "postgres", "PostgreSQL", "h2", "H2",
        "redis", "Redis", "elasticsearch", "Elasticsearch"
    ));
    
    /**
     * 主分词方法：使用IK分词器提取关键词
     * @param query 用户查询
     * @return 关键词数组
     */
    public String[] tokenize(String query) {
        if (query == null || query.trim().isEmpty()) {
            return new String[0];
        }
        
        List<String> keywords = new ArrayList<>();
        
        // 1. IK分词
        List<String> ikTokens = tokenizeWithIK(query);
        keywords.addAll(ikTokens);
        
        // 2. 匹配自定义术语（保留完整术语）
        List<String> customMatches = matchCustomTerms(query);
        keywords.addAll(customMatches);
        
        // 3. 过滤、去重、限制数量
        return keywords.stream()
            .filter(token -> token != null && !token.isEmpty())
            .filter(token -> token.length() >= 2)  // 过滤单字
            .filter(token -> !STOP_WORDS.contains(token.toLowerCase()))  // 过滤停用词
            .map(String::toLowerCase)  // 统一小写
            .distinct()
            .limit(10)  // 最多10个关键词
            .toArray(String[]::new);
    }
    
    /**
     * 使用IK分词器进行分词
     */
    private List<String> tokenizeWithIK(String text) {
        List<String> tokens = new ArrayList<>();
        
        try (StringReader reader = new StringReader(text)) {
            IKSegmenter ikSegmenter = new IKSegmenter(reader, USE_SMART_MODE);
            Lexeme lexeme;
            
            while ((lexeme = ikSegmenter.next()) != null) {
                String token = lexeme.getLexemeText();
                if (token != null && !token.isEmpty()) {
                    tokens.add(token);
                }
            }
        } catch (IOException e) {
            log.error("IK分词失败: {}", e.getMessage());
        }
        
        return tokens;
    }
    
    /**
     * 匹配自定义术语
     */
    private List<String> matchCustomTerms(String text) {
        List<String> matched = new ArrayList<>();
        String lowerText = text.toLowerCase();
        
        for (String term : CUSTOM_TERMS) {
            if (lowerText.contains(term.toLowerCase())) {
                matched.add(term);
            }
        }
        
        return matched;
    }
}
