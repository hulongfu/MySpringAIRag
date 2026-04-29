package com.myspringairag.configuration;

import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.scheduling.concurrent.ThreadPoolTaskExecutor;

import java.util.concurrent.Executor;

@Configuration
public class AsyncConfig {

    @Bean("documentProcessingExecutor")
    public Executor documentProcessingExecutor() {
        ThreadPoolTaskExecutor executor = new ThreadPoolTaskExecutor();
        executor.setCorePoolSize(1);  // 核心线程数=1，保证串行处理
        executor.setMaxPoolSize(1);   // 最大线程数=1
        executor.setQueueCapacity(10);
        executor.setThreadNamePrefix("doc-process-");
        executor.initialize();
        return executor;
    }
}
