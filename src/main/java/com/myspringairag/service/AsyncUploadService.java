package com.myspringairag.service;

import com.myspringairag.controller.SseController;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.scheduling.annotation.Async;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.UUID;
import java.util.concurrent.Semaphore;

@Slf4j
@Service
public class AsyncUploadService {

    @Value("${app.upload-dir}")
    private String uploadDir;

    @Autowired
    private SseController sseController;

    @Autowired
    private RagService ragService;

    // 信号量：只允许1个并发上传任务
    private final Semaphore uploadSemaphore = new Semaphore(1);

    /**
     * 尝试获取上传许可
     */
    public boolean tryAcquire() {
        return uploadSemaphore.tryAcquire();
    }

    /**
     * 提交异步上传任务（使用前端传来的taskId）
     */
    public void submitUploadWithTaskId(MultipartFile file, String taskId) throws IOException {
        // 1. 确保uploads目录存在
        Path uploadPath = Paths.get(uploadDir);
        Files.createDirectories(uploadPath);
        
        // 2. 保存临时文件到uploads目录（使用时间戳避免重名）
        String originalFilename = file.getOriginalFilename();
        String timestamp = String.valueOf(System.currentTimeMillis());
        String tempFilename = "temp_" + timestamp + "_" + originalFilename;
        Path tempFile = uploadPath.resolve(tempFilename);
        file.transferTo(tempFile.toFile());
        
        log.debug("Temp file saved to: {}", tempFile);
        
        // 3. 提交异步任务（使用前端传来的taskId）
        processDocument(taskId, tempFile, originalFilename);
        
        log.info("Upload task submitted: {} for file: {}", taskId, originalFilename);
    }

    /**
     * 提交异步上传任务（旧方法，保留兼容）
     */
    public String submitUpload(MultipartFile file) throws IOException {
        // 1. 确保uploads目录存在
        Path uploadPath = Paths.get(uploadDir);
        Files.createDirectories(uploadPath);
        
        // 2. 保存临时文件到uploads目录（使用时间戳避免重名）
        String originalFilename = file.getOriginalFilename();
        String timestamp = String.valueOf(System.currentTimeMillis());
        String tempFilename = "temp_" + timestamp + "_" + originalFilename;
        Path tempFile = uploadPath.resolve(tempFilename);
        file.transferTo(tempFile.toFile());
        
        log.debug("Temp file saved to: {}", tempFile);
        
        // 3. 生成任务ID
        String taskId = UUID.randomUUID().toString();
        
        // 4. 提交异步任务
        processDocument(taskId, tempFile, originalFilename);
        
        log.info("Upload task submitted: {} for file: {}", taskId, originalFilename);
        return taskId;
    }

    /**
     * 异步处理文档
     */
    @Async("documentProcessingExecutor")
    public void processDocument(String taskId, Path tempFile, String filename) {
        log.info("Starting async processing for task: {}, file: {}", taskId, filename);
        
        try {
            sseController.notifyProgress(taskId, 10, "正在读取文件...");
            
            sseController.notifyProgress(taskId, 30, "正在解析文档...");
            
            // 执行实际的上传逻辑
            ragService.uploadDocumentFromPath(tempFile, filename);
            
            sseController.notifyProgress(taskId, 90, "正在生成向量索引...");
            
            sseController.notifyCompletion(taskId, "文档处理完成！");
            
        } catch (Exception e) {
            log.error("Task {} failed", taskId, e);
            sseController.notifyCompletion(taskId, "处理失败: " + e.getMessage());
        } finally {
            // 清理临时文件
            try {
                Files.deleteIfExists(tempFile);
                log.debug("Temp file deleted: {}", tempFile);
            } catch (IOException e) {
                log.warn("Failed to delete temp file: {}", tempFile, e);
            }
            
            // 释放信号量
            uploadSemaphore.release();
            log.info("Task {} finished, semaphore released", taskId);
        }
    }
}
