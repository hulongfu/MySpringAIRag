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
     * 配合 AsyncUploadService 的信号量使用，实现严格的串行上传控制
     */
    @Bean("documentProcessingExecutor")
    public Executor documentProcessingExecutor() {
        ThreadPoolTaskExecutor executor = new ThreadPoolTaskExecutor();
        executor.setCorePoolSize(1);  // 核心线程数=1，保证串行处理
        executor.setMaxPoolSize(1);   // 最大线程数=1
        executor.setQueueCapacity(1); // 队列容量=1，与信号量逻辑对齐（最多1个任务等待）
        executor.setThreadNamePrefix("doc-process-");
        executor.initialize();
        return executor;
    }
    
    /**
     * 线程池监控调度器（独立Bean，由Spring管理生命周期）
     */
    @Bean
    public java.util.concurrent.ScheduledExecutorService threadPoolMonitorExecutor() {
        java.util.concurrent.ScheduledExecutorService monitor = 
            Executors.newSingleThreadScheduledExecutor(r -> {
                Thread t = new Thread(r, "thread-pool-monitor");
                t.setDaemon(true);
                return t;
            });
        
        log.info("Thread pool monitor executor initialized");
        return monitor;
    }
    
    /**
     * 并行计算线程池（用于Embedding生成、向量检索等CPU密集型任务）
     * 统一管理所有并行任务的线程资源
     */
    @Bean("parallelComputeExecutor")
    public ExecutorService parallelComputeExecutor(java.util.concurrent.ScheduledExecutorService monitorExecutor) {
        // CPU密集型任务：线程数 = CPU核心数
        int poolSize = Runtime.getRuntime().availableProcessors();
        
        log.info("Initializing parallel compute thread pool with {} threads (CPU cores: {})", 
            poolSize, Runtime.getRuntime().availableProcessors());
        
        // 使用有界队列，防止OOM
        java.util.concurrent.BlockingQueue<Runnable> queue = 
            new java.util.concurrent.ArrayBlockingQueue<>(100);  // 最多100个待处理任务
        
        // 使用原子计数器为线程编号，确保每个线程有唯一名称
        java.util.concurrent.atomic.AtomicInteger threadNumber = new java.util.concurrent.atomic.AtomicInteger(1);
        
        java.util.concurrent.ThreadPoolExecutor executor = new java.util.concurrent.ThreadPoolExecutor(
            poolSize,           // 核心线程数
            poolSize,           // 最大线程数（与核心相同，固定大小）
            60L,                // 空闲线程存活时间
            java.util.concurrent.TimeUnit.SECONDS,
            queue,              // 有界队列
            r -> {
                Thread thread = new Thread(r);
                thread.setName("parallel-compute-worker-" + threadNumber.getAndIncrement());
                thread.setDaemon(true);
                return thread;
            },
            new java.util.concurrent.ThreadPoolExecutor.CallerRunsPolicy()  // 队列满时由调用线程执行
        );
        
        // 添加监控：定期打印线程池状态（使用Spring管理的监控线程池）
        monitorExecutor.scheduleAtFixedRate(() -> {
            log.debug("ThreadPool Status: active={}, queue={}, completed={}",
                executor.getActiveCount(),
                queue.size(),
                executor.getCompletedTaskCount());
        }, 30, 30, java.util.concurrent.TimeUnit.SECONDS);
        
        return executor;
    }
}
