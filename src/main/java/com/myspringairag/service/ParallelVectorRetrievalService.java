package com.myspringairag.service;

import com.myspringairag.model.ScoredDocument;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.util.*;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.stream.Collectors;

/**
 * 并行向量检索服务 - 支持多查询变体并行检索 + RRF融合
 */
@Slf4j
@Service
public class ParallelVectorRetrievalService {
    
    @Autowired
    private EmbeddingService embeddingService;
    
    @Autowired
    private JVectorService jVectorService;
    
    @Autowired
    private QueryRewriteService queryRewriteService;
    
    @Autowired
    private com.myspringairag.service.QueryTokenizationService tokenizationService;
    
    // 线程池（用于并行执行多个查询）
    private final ExecutorService executor = Executors.newFixedThreadPool(4);
    
    /**
     * 多查询变体并行检索
     * @param userQuery 用户原始问题
     * @param topK 返回的最大结果数
     * @return 带分数的文档列表（已按最终分数排序）
     */
    public List<ScoredDocument> parallelSearch(String userQuery, int topK) {
        long startTime = System.currentTimeMillis();
        
        // 1. 生成查询变体
        List<String> variants = generateVariants(userQuery);
        log.info("Generated {} query variants: {}", variants.size(), variants);
        
        // 2. 并行执行每个变体的检索
        List<CompletableFuture<Map<Long, Float>>> futures = variants.stream()
            .map(variant -> CompletableFuture.supplyAsync(() -> {
                try {
                    // 生成embedding
                    float[] vector = embeddingService.embed(variant);
                    // 向量检索，返回带分数的结果
                    return jVectorService.searchWithScores(vector, topK);
                } catch (Exception e) {
                    log.error("Search failed for variant: {}", variant, e);
                    return Collections.<Long, Float>emptyMap();
                }
            }, executor))
            .collect(Collectors.toList());
        
        // 3. 等待所有任务完成
        CompletableFuture.allOf(futures.toArray(new CompletableFuture[0])).join();
        
        // 4. 收集所有结果
        List<Map<Long, Float>> allResults = futures.stream()
            .map(CompletableFuture::join)
            .filter(result -> !result.isEmpty())
            .collect(Collectors.toList());
        
        // 5. RRF融合
        List<ScoredDocument> merged = mergeScoresWithRRF(allResults, topK);
        
        log.info("Parallel search completed in {}ms, merged to {} documents", 
            System.currentTimeMillis() - startTime, merged.size());
        
        return merged;
    }
    
    /**
     * 生成查询变体（根据查询长度动态调整变体数量）
     */
    private List<String> generateVariants(String originalQuery) {
        List<String> variants = new ArrayList<>();
        
        // 变体1: 原始查询
        variants.add(originalQuery);
        
        // 变体2: 查询重写后的版本
        String rewritten = queryRewriteService.rewrite(originalQuery);
        if (!rewritten.equals(originalQuery)) {
            variants.add(rewritten);
        }
        
        // 变体3: 提取核心实体/关键词
        String entities = extractEntities(originalQuery);
        if (!entities.isEmpty() && !variants.contains(entities)) {
            variants.add(entities);
        }
        
        // 变体4: 简化版（只保留核心名词）
        String simplified = simplifyQuery(originalQuery);
        if (!simplified.isEmpty() && !variants.contains(simplified)) {
            variants.add(simplified);
        }
        
        // 变体5: 扩展关键词组合（增加同义词和相关词）
        List<String> expanded = expandKeywords(originalQuery);
        for (String exp : expanded) {
            if (!exp.isEmpty() && !variants.contains(exp)) {
                variants.add(exp);
            }
        }
        
        // 根据查询长度动态限制变体数量
        int maxVariants = getOptimalVariantCount(originalQuery);
        List<String> result = variants.stream().distinct().limit(maxVariants).collect(Collectors.toList());
        return result;
    }
    
    /**
     * 根据查询长度确定最优变体数量
     * - 简单查询（<10个字）：3个变体
     * - 中等查询（10-20个字）：5个变体
     * - 复杂查询（>20个字）：6个变体
     */
    private int getOptimalVariantCount(String query) {
        int length = query.length();
        if (length < 10) {
            return 3;
        } else if (length <= 20) {
            return 5;
        } else {
            return 6;
        }
    }
    
    /**
     * 提取核心实体/关键词（使用IK分词器 + 去重 + 合并相似词）
     * 例如："华为云桌面的docker中安装ollama" → "华为云 云桌面 docker ollama 安装"
     */
    private String extractEntities(String query) {
        // 使用IK分词器提取关键词
        String[] keywords = tokenizationService.tokenize(query);
                
        if (keywords.length == 0) {
            return query;
        }
                
        // 过滤掉单字符、停用词，并去重
        Set<String> uniqueKeywords = new LinkedHashSet<>();  // 使用LinkedHashSet保持顺序并去重
        for (String keyword : keywords) {
            String trimmed = keyword.trim();
            if (trimmed.length() > 1) {  // 保留长度>1的关键词
                uniqueKeywords.add(trimmed);  // 自动去重
            }
        }
                
        if (uniqueKeywords.isEmpty()) {
            return query;
        }
            
        // 优化：合并包含关系的词汇，避免冗余
        List<String> optimizedKeywords = mergeRelatedTerms(new ArrayList<>(uniqueKeywords));
                
        // 组合成查询字符串
        return String.join(" ", optimizedKeywords);
    }
        
    /**
     * 合并相关词汇，避免冗余
     * 例如：["华为云", "云桌面", "桌面"] → ["华为云", "云桌面"] （移除单独的"桌面"）
     */
    private List<String> mergeRelatedTerms(List<String> keywords) {
        if (keywords.size() <= 1) {
            return keywords;
        }
            
        // 按长度降序排序，优先保留长词
        keywords.sort((a, b) -> b.length() - a.length());
            
        List<String> result = new ArrayList<>();
        Set<String> covered = new HashSet<>();
            
        for (String keyword : keywords) {
            // 检查是否已经被更长的词覆盖
            boolean isCovered = false;
            for (String existing : result) {
                if (existing.contains(keyword)) {
                    isCovered = true;
                    break;
                }
            }
                
            if (!isCovered && !covered.contains(keyword)) {
                result.add(keyword);
                // 标记该词的所有子串为已覆盖
                for (int i = 2; i < keyword.length(); i++) {
                    for (int j = 0; j <= keyword.length() - i; j++) {
                        covered.add(keyword.substring(j, j + i));
                    }
                }
            }
        }
            
        return result;
    }
    
    /**
     * 简化查询（只保留核心名词）
     * 例如："启动open-webui的两种方式是什么？" → "open-webui 启动 方式"
     */
    private String simplifyQuery(String query) {
        // 去除所有疑问词和助词
        String simplified = query
            .replaceAll("^(如何在|怎么|怎样|如何|请问|帮我|教我|什么是|是什么|两种方式|的方法|的步骤|有哪些|包括哪些|具体是)[\\s]*", "")
            .replaceAll("(是什么|是什么意思|有哪些|包括哪些|具体是)[\\s]*[？？]?$", "")
            // 只去除真正的停用词，保留有意义的词汇（如"方式"、"方法"、"步骤"、"桌面"、"界面"等）
            // 注意：移除了"面"字，因为它在"桌面"、"方面"、"界面"等词中有意义
            .replaceAll("[的了在是我有和就不人都一一个上也很到说要你会着没有看好自己这他她它们那些什么哪吗呢吧啊哦呀哇哎嗯嘛啦哟呗哈嘻呵嘿喂诶咦呃噢呦唉哼切呸啐嘘吁吆喝喊叫嚷吼啸鸣啼泣哭笑怒骂吵闹打架斗争战胜败赢输得失进退来回走跑跳飞游爬滚翻转旋绕圈环周期时分秒年月日天夜早晚晨昏昼夕朝暮春夏秋冬东西南北中左右前后上下里外内间旁边角顶底头尾首末初始终结开关启停动静快慢高低大小长短粗细宽窄厚薄深浅重轻多少全半整零空满实虚假对错好坏美丑善恶爱恨喜悲欢愁忧虑思想念忘记忆知识懂明白清楚糊涂迷惑疑问答解释说讲谈论评议判断定决择选挑捡拿取放置摆排列序顺逆正向位置点线体形状态势力量能量质性格品特征象现表示显露藏隐出入进出回去归返还送接收发传递交换替代更改变化生长成熟老死活存亡兴衰盛败荣辱贵贱贫富穷通达阻塞堵通畅顺利害益损增减加乘除等于远近新旧古今中外土洋]", " ")
            .replaceAll("[?？。！，,、；:：]", " ")
            .replaceAll("\\s+", " ")
            .trim();
        
        // 如果简化后太短，返回原查询的核心部分
        if (simplified.length() < 3) {
            // 尝试提取英文单词或数字
            java.util.regex.Pattern pattern = java.util.regex.Pattern.compile("[a-zA-Z0-9_-]+");
            java.util.regex.Matcher matcher = pattern.matcher(query);
            StringBuilder entities = new StringBuilder();
            while (matcher.find()) {
                if (entities.length() > 0) {
                    entities.append(" ");
                }
                entities.append(matcher.group());
            }
            if (entities.length() > 0) {
                return entities.toString();
            }
        }
        
        return simplified;
    }
    
    /**
     * 扩展关键词组合（添加同义词和相关词）
     * 例如：“启动open-webui的两种方式是什么？” → ["open-webui serve", "open-webui run", "webui 启动"]
     */
    private List<String> expandKeywords(String query) {
        List<String> expanded = new ArrayList<>();
            
        // 提取核心技术名词
        java.util.regex.Pattern pattern = java.util.regex.Pattern.compile("[a-zA-Z][a-zA-Z0-9_-]*");
        java.util.regex.Matcher matcher = pattern.matcher(query);
        List<String> techTerms = new ArrayList<>();
        while (matcher.find()) {
            String term = matcher.group().toLowerCase();
            // 过滤掉常见的非技术词汇
            if (!term.matches("^(what|how|why|when|where|who|the|is|are|was|were|be|been|being|have|has|had|do|does|did|will|would|could|should|may|might|can|shall)$")) {
                techTerms.add(term);
            }
        }
            
        if (techTerms.isEmpty()) {
            return expanded;
        }
            
        // 检测查询意图，添加相关动词
        boolean isInstall = query.contains("安装") || query.contains("部署") || query.contains("配置");
        boolean isStart = query.contains("启动") || query.contains("运行") || query.contains("开始");
        boolean isUsage = query.contains("使用") || query.contains("方法") || query.contains("教程");
            
        // 如果有多个技术词，构建组合
        if (techTerms.size() >= 2) {
            // 保留所有技术词的组合（最重要）
            String allTerms = String.join(" ", techTerms);
                
            if (isInstall) {
                expanded.add(allTerms + " install");
                expanded.add(allTerms + " 安装");
            }
                
            if (isStart) {
                expanded.add(allTerms + " start");
                expanded.add(allTerms + " 启动");
            }
                
            if (isUsage) {
                expanded.add(allTerms + " usage");
                expanded.add(allTerms + " 使用方法");
            }
                
            // 也添加不带动词的纯组合
            if (!expanded.contains(allTerms)) {
                expanded.add(allTerms);
            }
        } else {
            // 只有一个技术词的情况
            String mainTerm = techTerms.get(0);
                
            if (isInstall) {
                expanded.add(mainTerm + " install");
                expanded.add(mainTerm + " setup");
                expanded.add(mainTerm + " deploy");
            }
                
            if (isStart) {
                expanded.add(mainTerm + " serve");
                expanded.add(mainTerm + " run");
                expanded.add(mainTerm + " start");
            }
                
            if (isUsage) {
                expanded.add(mainTerm + " usage");
                expanded.add(mainTerm + " tutorial");
                expanded.add(mainTerm + " guide");
            }
        }
            
        return expanded.stream().distinct().limit(4).collect(Collectors.toList());
    }
    
    /**
     * RRF融合多个查询变体的结果
     */
    private List<ScoredDocument> mergeScoresWithRRF(List<Map<Long, Float>> allResults, int topK) {
        Map<Long, Double> rrfScores = new HashMap<>();
        Map<Long, Float> maxOriginalScores = new HashMap<>();
        Set<Long> allDocIds = new HashSet<>();
        
        double K = 60.0;  // RRF常数
        
        // 遍历每个变体的结果
        for (int variantIdx = 0; variantIdx < allResults.size(); variantIdx++) {
            Map<Long, Float> variantResults = allResults.get(variantIdx);
            
            // 按分数排序，计算排名
            List<Map.Entry<Long, Float>> sortedEntries = variantResults.entrySet().stream()
                .sorted(Map.Entry.<Long, Float>comparingByValue().reversed())
                .collect(Collectors.toList());
            
            for (int rank = 0; rank < sortedEntries.size(); rank++) {
                Map.Entry<Long, Float> entry = sortedEntries.get(rank);
                Long docId = entry.getKey();
                Float score = entry.getValue();
                
                allDocIds.add(docId);
                
                // RRF分数累加
                double rrfScore = 1.0 / (K + rank + 1);
                rrfScores.merge(docId, rrfScore, Double::sum);
                
                // 保留最高原始分数
                maxOriginalScores.merge(docId, score, Math::max);
            }
        }
        
        // 计算最终分数 = 原始分数 * (1 + RRF分数 * 10)
        List<ScoredDocument> scoredDocs = allDocIds.stream()
            .map(docId -> {
                float originalScore = maxOriginalScores.get(docId);
                double rrfScore = rrfScores.getOrDefault(docId, 0.0);
                double finalScore = originalScore * (1 + rrfScore * 10);
                
                return new ScoredDocument(docId, null, originalScore, rrfScore, finalScore);
            })
            .sorted(Comparator.comparingDouble(ScoredDocument::getFinalScore).reversed())
            .limit(topK)
            .collect(Collectors.toList());
        
        return scoredDocs;
    }
}
