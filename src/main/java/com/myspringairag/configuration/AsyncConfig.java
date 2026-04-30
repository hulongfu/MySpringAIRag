package com.myspringairag.configuration;

import lombok.extern.slf4j.Slf4j;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.scheduling.concurrent.ThreadPoolTaskExecutor;

import java.util.concurrent.Executor;
import java.util.concurrent.Executors;
import java.util.concurrent.ExecutorService;

@Slf4j
@Configuration
public class AsyncConfig {

    /**
     * 文档处理线程池（单线程，保证串行处理）
     */
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
    
    /**
     * 并行计算线程池（用于Embedding生成、向量检索等CPU密集型任务）
     * 统一管理所有并行任务的线程资源
     */
    @Bean("parallelComputeExecutor")
    public ExecutorService parallelComputeExecutor() {
        // CPU密集型任务：线程数 = CPU核心数
        int poolSize = Runtime.getRuntime().availableProcessors();
        
        log.info("Initializing parallel compute thread pool with {} threads (CPU cores: {})", 
            poolSize, Runtime.getRuntime().availableProcessors());
        
        // 使用有界队列，防止OOM
        java.util.concurrent.BlockingQueue<Runnable> queue = 
            new java.util.concurrent.ArrayBlockingQueue<>(100);  // 最多100个待处理任务
        
        java.util.concurrent.ThreadPoolExecutor executor = new java.util.concurrent.ThreadPoolExecutor(
            poolSize,           // 核心线程数
            poolSize,           // 最大线程数（与核心相同，固定大小）
            60L,                // 空闲线程存活时间
            java.util.concurrent.TimeUnit.SECONDS,
            queue,              // 有界队列
            r -> {
                Thread thread = new Thread(r);
                thread.setName("parallel-compute-worker");
                thread.setDaemon(true);
                return thread;
            },
            new java.util.concurrent.ThreadPoolExecutor.CallerRunsPolicy()  // 队列满时由调用线程执行
        );
        
        // 添加监控：定期打印线程池状态
        Executors.newSingleThreadScheduledExecutor(r -> {
            Thread t = new Thread(r, "thread-pool-monitor");
            t.setDaemon(true);
            return t;
        }).scheduleAtFixedRate(() -> {
            log.debug("ThreadPool Status: active={}, queue={}, completed={}",
                executor.getActiveCount(),
                queue.size(),
                executor.getCompletedTaskCount());
        }, 30, 30, java.util.concurrent.TimeUnit.SECONDS);
        
        return executor;
    }
}
