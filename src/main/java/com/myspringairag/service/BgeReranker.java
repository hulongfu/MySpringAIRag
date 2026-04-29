package com.myspringairag.service;

import ai.djl.huggingface.tokenizers.HuggingFaceTokenizer;
import ai.onnxruntime.*;
import java.nio.file.Paths;
import java.util.HashMap;
import java.util.Map;

public class BgeReranker {

    private final OrtEnvironment env;
    private final OrtSession session;
    private final HuggingFaceTokenizer tokenizer;

    public BgeReranker(String modelDir) throws Exception {
        // 1. 初始化 ONNX Runtime
        this.env = OrtEnvironment.getEnvironment();
        OrtSession.SessionOptions options = new OrtSession.SessionOptions();
        this.session = env.createSession(
                Paths.get(modelDir, "onnx/model_fp16.onnx").toString(),
                options
        );

        // 2. 加载 tokenizer（自动读取 modelDir 下的 tokenizer.json 等配置）
        this.tokenizer = HuggingFaceTokenizer.builder()
                .optTokenizerPath(Paths.get(modelDir))
                .optMaxLength(512)
                .optPadToMaxLength()
                .optTruncation(true)
                .build();
    }

    /**
     * 对 query-passage 对进行重排序打分
     * @param query 查询文本
     * @param passage 候选文档
     * @return 相关性分数 (0.0 ~ 1.0)
     */
    public float rerank(String query, String passage) throws Exception {
        // 1. Tokenize：BGE-Reranker-v2-m3 需要 [CLS] query [SEP] passage [SEP] 格式
        var encoding = tokenizer.encode(query, passage, true, true);

        long[] inputIds = encoding.getIds();
        long[] attentionMask = encoding.getAttentionMask();

        // 2. 包装成 batch=1 的二维数组 [1, seqLen]
        long[][] inputIdsBatch = new long[][]{inputIds};
        long[][] attentionMaskBatch = new long[][]{attentionMask};

        // 3. 创建 ONNX 输入张量（v2-m3 只需要2个输入）
        OnnxTensor inputIdsTensor = OnnxTensor.createTensor(env, inputIdsBatch);
        OnnxTensor attentionMaskTensor = OnnxTensor.createTensor(env, attentionMaskBatch);

        // 4. 构建输入映射（v2-m3 不需要 token_type_ids）
        Map<String, OnnxTensor> inputs = new HashMap<>();
        inputs.put("input_ids", inputIdsTensor);
        inputs.put("attention_mask", attentionMaskTensor);

        // 5. 推理
        OrtSession.Result results = session.run(inputs);

        // 6. 获取分数并应用 sigmoid 激活
        // BGE-Reranker-v2-m3 输出原始 logits，需要转换为概率
        float[][] logits = (float[][]) results.get(0).getValue();
        float logit = logits[0][0];
        
        // 应用 sigmoid: 1 / (1 + exp(-x))
        float score = (float) (1.0 / (1.0 + Math.exp(-logit)));
        return score;
    }

    /**
     * 批量重排序
     */
    public Map<String, Float> rerankBatch(String query, String[] passages) throws Exception {
        Map<String, Float> scores = new HashMap<>();
        for (String passage : passages) {
            float score = rerank(query, passage);
            scores.put(passage, score);
        }
        return scores.entrySet().stream()
                .sorted(Map.Entry.<String, Float>comparingByValue().reversed())
                .collect(java.util.LinkedHashMap::new,
                        (m, e) -> m.put(e.getKey(), e.getValue()),
                        java.util.LinkedHashMap::putAll);
    }

    public void close() {
        try {
            session.close();
            env.close();
            tokenizer.close();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public static void main(String[] args) throws Exception {
        String modelDir = "D:/ideaSpace/MyPython/models/bge-reranker-v2-m3-ONNX";

        try {
            BgeReranker reranker = new BgeReranker(modelDir);
            // 单条测试
            float score = reranker.rerank(
                    "什么是Spring AI",
                    "Spring AI 是 Spring 框架的 AI 集成项目，提供大模型应用开发支持"
            );
            System.out.println("相关性分数: " + score);

            // 批量重排序测试
            String query = "Java 嵌入式数据库";
            String[] candidates = {
                    "H2 是一个用 Java 编写的嵌入式关系型数据库",
                    "Redis 是一个内存键值存储系统",
                    "MySQL 是最流行的开源关系型数据库"
            };

            Map<String, Float> ranked = reranker.rerankBatch(query, candidates);
            System.out.println("\n重排序结果：");
            ranked.forEach((doc, s) -> System.out.printf("%.4f  %s%n", s, doc));
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}