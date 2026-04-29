package com.myspringairag.controller;

import lombok.extern.slf4j.Slf4j;
import org.springframework.http.MediaType;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.servlet.mvc.method.annotation.SseEmitter;

import java.io.IOException;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

@Slf4j
@RestController
@RequestMapping("/api/knowledge")
public class SseController {

    private final Map<String, SseEmitter> emitters = new ConcurrentHashMap<>();

    /**
     * 前端连接 SSE
     */
    @GetMapping(value = "/stream/{taskId}", produces = MediaType.TEXT_EVENT_STREAM_VALUE)
    public SseEmitter stream(@PathVariable String taskId) {
        // 0L 表示无超时，连接会一直保持直到手动关闭
        SseEmitter emitter = new SseEmitter(0L);
        
        emitters.put(taskId, emitter);
        log.info("SSE connection established for task: {}", taskId);
        
        // 连接完成时移除
        emitter.onCompletion(() -> {
            emitters.remove(taskId);
            log.info("SSE connection completed for task: {}", taskId);
        });
        
        // 超时时移除（虽然设置了0L，但网络断开等可能触发）
        emitter.onTimeout(() -> {
            emitters.remove(taskId);
            log.warn("SSE connection timed out for task: {}", taskId);
        });
        
        // 错误时移除
        emitter.onError(e -> {
            emitters.remove(taskId);
            log.error("SSE connection error for task: {}", taskId, e);
        });
        
        return emitter;
    }
    
    /**
     * 任务完成时推送消息
     */
    public void notifyCompletion(String taskId, String message) {
        SseEmitter emitter = emitters.get(taskId);
        if (emitter != null) {
            try {
                emitter.send(SseEmitter.event()
                    .name("progress")
                    .data(Map.of(
                        "status", "COMPLETED",
                        "message", message,
                        "progress", 100,
                        "timestamp", System.currentTimeMillis()
                    ))
                );
                emitter.complete();  // 关闭连接
                log.info("Task {} completed notification sent", taskId);
            } catch (IOException e) {
                log.error("Failed to send completion event for task: {}", taskId, e);
                emitter.completeWithError(e);
            }
        } else {
            // SSE连接尚未建立，等待后重试
            log.warn("No SSE emitter found for task: {}, will retry in 100ms", taskId);
            try {
                Thread.sleep(100);
                emitter = emitters.get(taskId);
                if (emitter != null) {
                    emitter.send(SseEmitter.event()
                        .name("progress")
                        .data(Map.of(
                            "status", "COMPLETED",
                            "message", message,
                            "progress", 100,
                            "timestamp", System.currentTimeMillis()
                        ))
                    );
                    emitter.complete();
                    log.info("Task {} completed notification sent (retry success)", taskId);
                } else {
                    log.error("Still no SSE emitter for task: {} after retry", taskId);
                }
            } catch (Exception ex) {
                log.error("Retry failed for task: {}", taskId, ex);
            }
        }
    }
    
    /**
     * 推送进度
     */
    public void notifyProgress(String taskId, int progress, String status) {
        SseEmitter emitter = emitters.get(taskId);
        if (emitter != null) {
            try {
                emitter.send(SseEmitter.event()
                    .name("progress")
                    .data(Map.of(
                        "progress", progress,
                        "status", status,
                        "timestamp", System.currentTimeMillis()
                    ))
                );
            } catch (IOException e) {
                log.error("Failed to send progress event for task: {}", taskId, e);
                emitter.completeWithError(e);
            }
        }
    }
}
