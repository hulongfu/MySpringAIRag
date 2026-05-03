package com.myspringairag.controller;

import com.myspringairag.model.Document;
import com.myspringairag.service.AsyncUploadService;
import com.myspringairag.service.RagService;
import lombok.Data;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.io.IOException;

@Slf4j
@RestController
@RequestMapping("/api")
public class RagController {
    
    private final RagService ragService;
    
    @Autowired
    private AsyncUploadService asyncUploadService;
    
    public RagController(RagService ragService) {
        this.ragService = ragService;
    }
    
    /**
     * 上传文档（支持异步处理）
     */
    @PostMapping("/upload")
    public ResponseEntity<Map<String, Object>> uploadDocument(
            @RequestParam("file") MultipartFile file,
            @RequestParam("taskId") String taskId) {
        Map<String, Object> response = new HashMap<>();
        
        try {
            if (file.isEmpty()) {
                response.put("success", false);
                response.put("message", "文件不能为空");
                return ResponseEntity.badRequest().body(response);
            }
            
            long fileSize = file.getSize();
            
            // 1. 检查文件大小
            if (fileSize > 50 * 1024 * 1024) {
                response.put("success", false);
                response.put("message", "文件大小超过50MB限制，请拆分文件或联系管理员");
                return ResponseEntity.badRequest().body(response);
            }
            
            // 2. 大文件警告
            if (fileSize > 10 * 1024 * 1024) {
                log.warn("Large file upload: {} ({} MB)", file.getOriginalFilename(), fileSize / 1024 / 1024);
            }
            
            // 3. 提交异步任务（使用前端传来的taskId）
            // 信号量管理已移至 Service 层内部，确保异常时也能正确释放
            asyncUploadService.submitUploadWithTaskId(file, taskId);
            
            response.put("success", true);
            response.put("message", "文件已提交处理，请通过SSE监听进度");
            response.put("taskId", taskId);
            response.put("filename", file.getOriginalFilename());
            
            return ResponseEntity.ok(response);
            
        } catch (IllegalStateException e) {
            // 信号量获取失败（系统繁忙）
            log.warn("Upload rejected due to concurrent limit: {}", e.getMessage());
            response.put("success", false);
            response.put("message", e.getMessage());
            return ResponseEntity.status(429).body(response);  // 429 Too Many Requests
            
        } catch (Exception e) {
            log.error("Upload failed", e);
            response.put("success", false);
            response.put("message", "上传失败: " + e.getMessage());
            return ResponseEntity.internalServerError().body(response);
        }
    }
    
    /**
     * 问答接口（同步）
     */
    @PostMapping("/ask")
    public ResponseEntity<Map<String, Object>> askQuestion(@RequestBody QuestionRequest request) {
        Map<String, Object> response = new HashMap<>();
        
        try {
            if (request.getQuestion() == null || request.getQuestion().trim().isEmpty()) {
                response.put("success", false);
                response.put("message", "问题不能为空");
                return ResponseEntity.badRequest().body(response);
            }
            
            String answer = ragService.answerQuestion(request.getQuestion());
            
            response.put("success", true);
            response.put("answer", answer);
            
            return ResponseEntity.ok(response);
            
        } catch (Exception e) {
            log.error("Question answering failed", e);
            response.put("success", false);
            response.put("message", "回答失败: " + e.getMessage());
            return ResponseEntity.internalServerError().body(response);
        }
    }
    
    /**
     * 流式问答接口（SSE）
     */
    @PostMapping(value = "/ask/stream", produces = org.springframework.http.MediaType.TEXT_EVENT_STREAM_VALUE)
    public org.springframework.web.servlet.mvc.method.annotation.SseEmitter askQuestionStream(@RequestBody QuestionRequest request) {
        // 创建SSE emitter，设置超时时间为0（不超时）
        org.springframework.web.servlet.mvc.method.annotation.SseEmitter emitter = 
            new org.springframework.web.servlet.mvc.method.annotation.SseEmitter(0L);
        
        try {
            if (request.getQuestion() == null || request.getQuestion().trim().isEmpty()) {
                emitter.send(org.springframework.web.servlet.mvc.method.annotation.SseEmitter.event()
                    .name("error")
                    .data(Map.of("message", "问题不能为空")));
                emitter.complete();
                return emitter;
            }
            
            // 异步执行流式回答
            CompletableFuture.runAsync(() -> {
                try {
                    ragService.answerQuestionStream(request.getQuestion(), emitter);
                } catch (Exception e) {
                    log.error("Stream question answering failed", e);
                    try {
                        emitter.send(org.springframework.web.servlet.mvc.method.annotation.SseEmitter.event()
                            .name("error")
                            .data(Map.of("message", "回答失败: " + e.getMessage())));
                    } catch (IOException ioException) {
                        log.error("Failed to send error event", ioException);
                    }
                    emitter.completeWithError(e);
                }
            });
            
        } catch (Exception e) {
            log.error("Failed to create stream emitter", e);
            emitter.completeWithError(e);
        }
        
        return emitter;
    }
    
    /**
     * 获取所有文档列表
     */
    @GetMapping("/documents")
    public ResponseEntity<Map<String, Object>> getDocuments() {
        Map<String, Object> response = new HashMap<>();
        
        try {
            List<String> filenames = ragService.getAllDocuments();
            
            response.put("success", true);
            response.put("documents", filenames);
            response.put("count", filenames.size());
            
            return ResponseEntity.ok(response);
            
        } catch (Exception e) {
            log.error("Failed to get documents", e);
            response.put("success", false);
            response.put("message", "获取文档列表失败: " + e.getMessage());
            return ResponseEntity.internalServerError().body(response);
        }
    }
    
    /**
     * 删除文档
     */
    @DeleteMapping("/documents/{filename}")
    public ResponseEntity<Map<String, Object>> deleteDocument(@PathVariable String filename) {
        Map<String, Object> response = new HashMap<>();
        
        try {
            ragService.deleteDocument(filename);
            
            response.put("success", true);
            response.put("message", "文档删除成功");
            
            return ResponseEntity.ok(response);
            
        } catch (Exception e) {
            log.error("Delete failed", e);
            response.put("success", false);
            response.put("message", "删除失败: " + e.getMessage());
            return ResponseEntity.internalServerError().body(response);
        }
    }
    
    @Data
    public static class QuestionRequest {
        private String question;
    }
}
