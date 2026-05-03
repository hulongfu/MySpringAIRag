package com.myspringairag.service;

import com.myspringairag.controller.SseController;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Lazy;
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

    @Autowired
    @Lazy
    private AsyncUploadService self; // 注入自身代理（使用@Lazy解决循环依赖）

    // 信号量：只允许1个并发上传任务
    private final Semaphore uploadSemaphore = new Semaphore(1);

    /**
     * 提交异步上传任务（使用前端传来的taskId）- 安全版本
     * 在方法内部管理信号量，确保异常时也能正确释放
     */
    public void submitUploadWithTaskId(MultipartFile file, String taskId) throws IOException {
        // 1. 尝试获取信号量许可
        if (!uploadSemaphore.tryAcquire()) {
            throw new IllegalStateException("当前有文件正在上传，请稍后再试");
        }
        
        try {
            // 2. 确保uploads目录存在
            Path uploadPath = Paths.get(uploadDir);
            Files.createDirectories(uploadPath);
            
            // 3. 保存临时文件到uploads目录（使用时间戳避免重名）
            String originalFilename = file.getOriginalFilename();
            
            // 安全检查：文件名不能为空
            if (originalFilename == null || originalFilename.trim().isEmpty()) {
                throw new IllegalArgumentException("Filename cannot be empty");
            }
            
            // 安全处理：只保留文件名，去除路径信息（防止路径遍历攻击）
            String safeFilename = Paths.get(originalFilename).getFileName().toString();
            
            // 生成安全的临时文件名
            String timestamp = String.valueOf(System.currentTimeMillis());
            String tempFilename = "temp_" + timestamp + "_" + safeFilename;
            Path tempFile = uploadPath.resolve(tempFilename);
            
            // 额外检查：确保最终路径在 uploads 目录内
            if (!tempFile.normalize().startsWith(uploadPath.normalize())) {
                throw new IllegalArgumentException("Invalid filename: " + originalFilename);
            }
            
            file.transferTo(tempFile.toFile());
            
            // 4. 提交异步任务（使用前端传来的taskId）
            self.processDocument(taskId, tempFile, originalFilename);
            
            log.info("Upload task submitted: {} for file: {}", taskId, originalFilename);
            
        } catch (Exception e) {
            // 如果保存文件失败，必须立即释放信号量，否则服务将死锁
            uploadSemaphore.release();
            log.error("Failed to submit upload task: {}", taskId, e);
            throw e;
        }
    }

    /**
     * 异步处理文档
     */
    @Async("documentProcessingExecutor")
    public void processDocument(String taskId, Path tempFile, String filename) {
        log.info("Starting async processing for task: {}, file: {}", taskId, filename);
        
        try {
            // 设置当前任务ID（供 RagService 使用）
            RagService.setCurrentTaskId(taskId);
            
            // 执行文件读取和解析（包含分块、向量化、存储）
            // RagService 内部会在关键节点推送进度（10%, 30%, 60%, 90%）
            ragService.uploadDocumentFromPath(tempFile, filename);
            
            // 100% - 全部完成
            sseController.notifyCompletion(taskId, "文档处理完成！");
            
        } catch (Exception e) {
            log.error("Task {} failed", taskId, e);
            sseController.notifyCompletion(taskId, "处理失败: " + e.getMessage());
        } finally {
            // 清除任务ID
            RagService.clearCurrentTaskId();
            
            // 清理临时文件
            try {
                Files.deleteIfExists(tempFile);
            } catch (IOException e) {
                log.warn("Failed to delete temp file: {}", tempFile, e);
            }
            
            // 释放信号量
            uploadSemaphore.release();
        }
    }
}
