package com.myspringairag.configuration;

import com.myspringairag.service.BgeReranker;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

@Slf4j
@Configuration
public class RerankerConfiguration {
    
    @Value("${app.reranker-model-path:D:/ideaSpace/MyPython/models/bge-reranker-v2-m3-ONNX}")
    private String rerankerModelPath;
    
    @Bean
    public BgeReranker bgeReranker() {
        try {
            log.info("Initializing BGE Reranker from: {}", rerankerModelPath);
            return new BgeReranker(rerankerModelPath);
        } catch (Exception e) {
            log.error("Failed to initialize BGE Reranker", e);
            return null;
        }
    }
}
