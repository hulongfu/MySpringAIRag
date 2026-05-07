package com.myspringairag.service;

import com.github.benmanes.caffeine.cache.Cache;
import com.github.benmanes.caffeine.cache.Caffeine;
import io.github.jbellis.jvector.graph.*;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import jakarta.annotation.PostConstruct;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Service;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.TimeUnit;
import java.util.stream.Collectors;

import static com.myspringairag.util.VectorUtils.toVectorFloat;

/**
 * JVector向量服务 - 混合架构(内存热缓存 + 磁盘索引)
 * 
 * 架构设计:
 * 1. 热数据层:Caffeine缓存最近访问的向量(默认10000条)
 * 2. 持久化层:H2数据库存储所有向量数据
 * 3. 索引层:JVector HNSW图索引(启动时从数据库构建)
 * 
 * 优势:
 * - 避免全量向量常驻内存,防止OOM
 * - 热数据快速访问,冷数据按需加载
 * - 支持百万级向量数据存储
 */
@Slf4j
@Service
public class JVectorService {
    
    private final JdbcTemplate jdbcTemplate;
    
    @Value("${jvector.index-path}")
    private String indexPath;
    
    @Value("${jvector.dimensions}")
    private int dimensions;
    
    @Value("${jvector.top-k}")
    private int topK;
    
    @Value("${jvector.similarity-threshold}")
    private float similarityThreshold;
    
    // === 安全模式配置（防止OOM）===
    @Value("${jvector.safe-mode:true}")
    private boolean safeMode;
    
    @Value("${jvector.max-memory-ratio:0.5}")
    private double maxMemoryRatio;
    
    @Value("${jvector.gc-threshold:0.85}")
    private double gcThreshold;
    
    @Value("${jvector.emergency-gc-threshold:0.9}")
    private double emergencyGcThreshold;
    
    // JVector索引
    // volatile保证索引重建时的可见性
    private volatile GraphIndex graphIndex;

    // 映射:nodeId -> docId(用于索引节点到文档ID的转换)
    // volatile保证索引重建时的可见性，ConcurrentHashMap保证并发安全
    private volatile Map<Integer, Long> nodeIdToDocIdMap = new ConcurrentHashMap<>();
    
    // 反向映射:docId -> nodeId(用于快速查找文档在索引中的位置)
    // volatile保证索引重建时的可见性，ConcurrentHashMap保证并发安全
    private volatile Map<Long, Integer> docIdToNodeIdMap = new ConcurrentHashMap<>();
    
    // 热数据缓存:缓存最近访问的向量(避免频繁从数据库读取)
    // 默认缓存10000个向量,30分钟未访问则淘汰
    private final Cache<Long, float[]> hotVectorCache = Caffeine.newBuilder()
        .maximumSize(10_000)
        .expireAfterAccess(30, TimeUnit.MINUTES)
        .recordStats()
        .build();
    
    // 向量总数计数器（原子操作，线程安全）
    private final AtomicInteger totalVectorCount = new AtomicInteger(0);
    
    // 软删除集合:跟踪已标记删除但尚未从索引中移除的docId
    // 用于搜索时过滤，避免全量重建索引
    // 注意：每次删除操作后会立即触发清理（forceCleanupDeletedVectors），因此该集合通常为空或很小
    // 定时任务（每天凌晨2点）作为兜底机制，扫描孤儿数据并清理异常情况
    private final Set<Long> deletedDocIds = ConcurrentHashMap.newKeySet();
    
    public JVectorService(JdbcTemplate jdbcTemplate) {
        this.jdbcTemplate = jdbcTemplate;
    }
    
    @PostConstruct
    public void init() {
        log.info("Initializing JVector hybrid architecture (cache + disk index) with dimensions={}", dimensions);
        log.info("Safe mode: {} (max_memory_ratio={}, gc_threshold={}, emergency_threshold={})", 
            safeMode, maxMemoryRatio, gcThreshold, emergencyGcThreshold);
        
        try {
            // 创建索引目录
            Path indexDir = Paths.get(indexPath);
            Files.createDirectories(indexDir);
            
            // 从数据库加载或构建索引
            loadOrBuildIndex();
            
            log.info("JVector hybrid index initialized successfully with {} vectors", 
                     graphIndex != null ? graphIndex.size() : 0);
            log.info("Hot vector cache configured: max_size=10000, expire_after_access=30min");
            
        } catch (Exception e) {
            log.error("Failed to initialize JVector hybrid index", e);
            throw new RuntimeException("JVector initialization failed", e);
        }
    }
    
    /**
     * 从数据库加载数据并构建索引(三阶段:映射→临时全量→清理)
     */
    private void loadOrBuildIndex() {
        log.info("=== Phase 1: Loading vector metadata ===");
        
        // 检查vectors表是否存在
        String checkTableSql = "SELECT COUNT(*) FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_NAME = 'VECTORS'";
        Integer tableCount = jdbcTemplate.queryForObject(checkTableSql, Integer.class);
        
        if (tableCount == null || tableCount == 0) {
            log.info("Vectors table does not exist yet, skipping index build");
            return;
        }
        
        // 获取总数量
        String countSql = "SELECT COUNT(*) FROM vectors";
        Long totalCount = jdbcTemplate.queryForObject(countSql, Long.class);
        if (totalCount == null || totalCount == 0) {
            log.info("No vectors found in database");
            return;
        }
        
        log.info("Found {} vectors in database", totalCount);
        
        // 分批加载nodeId <-> docId映射关系(不加载向量数据)
        int batchSize = 5000;  // 映射很小,可以大批次
        long offset = 0;
        Map<Integer, Long> tempNodeIdMap = new HashMap<>();
        Map<Long, Integer> tempDocIdMap = new HashMap<>();
        int loadedCount = 0;
        
        while (offset < totalCount) {
            // 只查询id和doc_id,不查询vector_data
            String sql = "SELECT v.id, v.doc_id FROM vectors v ORDER BY v.id LIMIT ? OFFSET ?";
            List<Map<String, Object>> rows = jdbcTemplate.queryForList(sql, batchSize, offset);
            
            for (Map<String, Object> row : rows) {
                Long docId = ((Number) row.get("doc_id")).longValue();
                int nodeId = loadedCount++;
                tempNodeIdMap.put(nodeId, docId);
                tempDocIdMap.put(docId, nodeId);
            }
            
            offset += rows.size();
            if (loadedCount % 10000 == 0) {
                log.info("Loaded metadata: {}/{} vectors", loadedCount, totalCount);
            }
        }
        
        // 更新状态
        this.nodeIdToDocIdMap = tempNodeIdMap;
        this.docIdToNodeIdMap = tempDocIdMap;
        this.totalVectorCount.set(loadedCount);
        
        log.info("Metadata loaded successfully. Total vectors: {}", totalVectorCount.get());
        
        // === Phase 2: 临时全量加载向量用于构建索引 ===
        log.info("=== Phase 2: Loading all vectors to temporary storage for index building ===");
        List<float[]> tempVectors = loadAllVectorsTemporarily();
        
        log.info("All vectors loaded to temp storage. Building HNSW index...");
        
        // 用临时List构建索引
        buildIndexWithTempStorage(tempVectors);
        
        log.info("HNSW index built successfully.");
        
        // === Phase 3: 清理临时数据,释放内存 ===
        log.info("=== Phase 3: Clearing temporary storage ===");
        tempVectors.clear();
        tempVectors = null;  // 帮助GC回收
        
        // 保存索引到文件
        saveIndex();
        
        log.info("Temporary storage cleared. Index ready for use.");
        log.info("Hot cache is empty now, will populate on-demand during search");
        log.info("Cache configured: max_size=10000, expire_after_access=30min");
    }
    
    /**
     * 构建索引(使用RAVV包装器,向量从缓存/数据库动态获取)
     * 注意:此方法用于运行时搜索,不适用于启动时的索引构建
     */
    private void buildIndex() {
        // 创建RandomAccessVectorValues包装器(从缓存/数据库动态获取向量)
        RandomAccessVectorValues ravv = new RandomAccessVectorValues() {
            @Override
            public int size() {
                return totalVectorCount.get();
            }
            
            @Override
            public int dimension() {
                return dimensions;
            }
            
            @Override
            public VectorFloat<?> getVector(int i) {
                // 通过nodeId找到docId,然后从缓存或数据库获取向量
                Long docId = nodeIdToDocIdMap.get(i);
                if (docId == null) {
                    throw new IllegalStateException("Invalid nodeId: " + i);
                }
                
                // 先从缓存获取
                float[] vector = hotVectorCache.getIfPresent(docId);
                if (vector != null) {
                    return toVectorFloat(vector);
                }
                
                // 缓存未命中,从数据库加载
                vector = loadVectorFromDatabase(docId);
                if (vector != null) {
                    // 放入缓存
                    hotVectorCache.put(docId, vector);
                    return toVectorFloat(vector);
                }
                
                throw new IllegalStateException("Vector not found for docId: " + docId);
            }
            
            @Override
            public boolean isValueShared() {
                return false;
            }
            
            /**
             * 返回RAVV实例的副本，用于并发搜索。
             * 
             * 【安全性说明】
             * 当前实现直接返回this（自身引用），这是安全的，因为：
             * 1. 本RAVV是无状态的：只读取外部变量，不维护任何实例级可变状态
             * 2. nodeIdToDocIdMap：ConcurrentHashMap，线程安全，支持并发读写
             * 3. hotVectorCache：Caffeine缓存，内部使用ConcurrentHashMap，线程安全
             * 4. jdbcTemplate：Spring管理的JdbcTemplate，线程安全
             * 5. loadVectorFromDatabase()：每次调用创建局部变量，无共享状态
             * 
             * 所有写操作（cache.put）都委托给线程安全的Caffeine组件。
             * 因此，多线程共享此RAVV实例是安全的。
             * 
             * 【注意】
             * 如果未来修改添加了实例级可变状态（如本地缓存、计数器等），
             * 此方法必须更新为创建真正的深拷贝实例。
             */
            @Override
            public RandomAccessVectorValues copy() {
                return this;
            }
        };
        
        // 配置并构建图索引
        GraphIndexBuilder builder = new GraphIndexBuilder(
            ravv,
            VectorSimilarityFunction.COSINE,
            16,   // M - 图度数
            100,  // efConstruction
            1.2f, // overflow
            1.2f  // alpha
        );
        
        // 获取索引(build会自动从RAVV读取所有向量)
        this.graphIndex = builder.build(ravv);
    }
    
    /**
     * 使用临时向量列表构建索引(启动时使用,避免频繁DB查询)
     * @param tempVectors 临时存储的所有向量
     */
    private void buildIndexWithTempStorage(List<float[]> tempVectors) {
        // 创建RAVV包装器,直接从临时List读取向量
        RandomAccessVectorValues ravv = new RandomAccessVectorValues() {
            @Override
            public int size() {
                return tempVectors.size();
            }
            
            @Override
            public int dimension() {
                return dimensions;
            }
            
            @Override
            public VectorFloat<?> getVector(int i) {
                // 直接从临时List获取,超快!
                return toVectorFloat(tempVectors.get(i));
            }
            
            @Override
            public boolean isValueShared() {
                return false;
            }
            
            /**
             * 返回RAVV实例的副本，用于并发搜索。
             * 
             * 【安全性说明】
             * 当前实现直接返回this（自身引用），这是安全的，因为：
             * 1. 本RAVV是无状态的：只从tempVectors List中读取向量数据
             * 2. tempVectors：启动时构建索引期间使用的临时List，单线程访问
             * 3. dimensions：final常量，不可变
             * 
             * 此RAVV仅用于启动时的buildIndexWithTempStorage()方法，
             * 该方法在单线程环境中执行，不存在并发访问。
             * 
             * 【注意】
             * 虽然当前场景下单线程使用，但为了符合接口契约和防御性编程，
             * 保留copy()方法的正确实现。如果未来改为并发使用，
             * 需要评估是否需要创建独立副本。
             */
            @Override
            public RandomAccessVectorValues copy() {
                return this;
            }
        };
        
        // 配置并构建图索引
        GraphIndexBuilder builder = new GraphIndexBuilder(
            ravv,
            VectorSimilarityFunction.COSINE,
            16,   // M - 图度数
            100,  // efConstruction
            1.2f, // overflow
            1.2f  // alpha
        );
        
        // 构建索引
        this.graphIndex = builder.build(ravv);
    }
    
    /**
     * 保存索引到文件(当前实现为从数据库重建,无需文件持久化)
     */
    private void saveIndex() {
        // 索引在应用启动时从数据库重建,无需额外文件持久化
    }
    
    /**
     * 临时全量加载所有向量到List(仅用于启动时构建索引)
     * 
     * 【安全模式】当 jvector.safe-mode=true 时：
     * - 动态计算批次大小（基于可用内存）
     * - 实时监控内存使用率
     * - 自动触发 GC 防止 OOM
     * - 超过阈值时抛出友好异常
     * 
     * 【普通模式】当 jvector.safe-mode=false 时：
     * - 使用固定批次大小（1000）
     * - 不监控内存（性能略优，但有 OOM 风险）
     * 
     * @return 包含所有向量的List
     */
    private List<float[]> loadAllVectorsTemporarily() {
        if (safeMode) {
            log.info("=== Safe Mode ENABLED: Dynamic batch sizing with memory monitoring ===");
            return loadAllVectorsWithSafeMode();
        } else {
            log.info("=== Safe Mode DISABLED: Using fixed batch size (faster but risky) ===");
            return loadAllVectorsLegacyMode();
        }
    }
    
    /**
     * 安全模式：动态批次大小 + 内存监控
     */
    private List<float[]> loadAllVectorsWithSafeMode() {
        log.info("Loading vectors with dynamic batch sizing...");
        long startTime = System.currentTimeMillis();
        
        // 1. 获取 JVM 内存信息
        Runtime runtime = Runtime.getRuntime();
        long maxMemory = runtime.maxMemory();
        long usedMemory = runtime.totalMemory() - runtime.freeMemory();
        long availableMemory = maxMemory - usedMemory;
        
        // 2. 计算安全批次大小（预留 maxMemoryRatio 比例的内存给索引构建）
        long safeMemoryBudget = (long) (availableMemory * maxMemoryRatio);
        int vectorSizeBytes = dimensions * 4;  // float = 4 bytes
        int estimatedBatchSize = (int) (safeMemoryBudget / vectorSizeBytes);
        
        // 限制批次范围：最小 100，最大 5000
        int batchSize = Math.min(Math.max(estimatedBatchSize, 100), 5000);
        
        log.info("Memory stats - Max: {}MB, Used: {}MB, Available: {}MB", 
            maxMemory / 1024 / 1024, 
            usedMemory / 1024 / 1024, 
            availableMemory / 1024 / 1024);
        log.info("Calculated batch size: {} vectors (dimension={}, memory_ratio={})", 
            batchSize, dimensions, maxMemoryRatio);
        
        List<float[]> tempVectors = new ArrayList<>(totalVectorCount.get());
        long offset = 0;
        int loadedCount = 0;
        
        try {
            while (offset < totalVectorCount.get()) {
                // 3. 动态检查内存（每处理一个批次检查一次）
                if (loadedCount > 0 && loadedCount % batchSize == 0) {
                    checkAndManageMemory(runtime, maxMemory, loadedCount, totalVectorCount.get());
                }
                
                // 4. 加载批次
                String sql = "SELECT v.vector_data, v.dimension FROM vectors v ORDER BY v.id LIMIT ? OFFSET ?";
                List<Map<String, Object>> rows = jdbcTemplate.queryForList(sql, batchSize, offset);
                
                for (Map<String, Object> row : rows) {
                    byte[] vectorBytes = (byte[]) row.get("vector_data");
                    int dimension = ((Number) row.get("dimension")).intValue();
                    float[] vector = deserializeVector(vectorBytes, dimension);
                    tempVectors.add(vector);
                    loadedCount++;
                }
                
                offset += rows.size();
                
                // 5. 进度日志（每 10000 个向量输出一次）
                if (loadedCount % 10000 == 0) {
                    long currentUsed = runtime.totalMemory() - runtime.freeMemory();
                    double memoryUsagePercent = (double) currentUsed / maxMemory * 100;
                    log.info("Loaded {}/{} vectors (Memory usage: {}%)", 
                        loadedCount, totalVectorCount.get(),
                        String.format("%.1f", memoryUsagePercent));
                }
            }
        } catch (IOException e) {
            log.error("Failed to deserialize vectors", e);
            throw new RuntimeException("Vector deserialization failed", e);
        }
        
        long duration = System.currentTimeMillis() - startTime;
        log.info("All vectors loaded in safe mode: {} vectors in {}ms", 
            tempVectors.size(), duration);
        
        return tempVectors;
    }
    
    /**
     * 普通模式：固定批次大小（原有逻辑）
     */
    private List<float[]> loadAllVectorsLegacyMode() {
        log.info("Loading all vectors to temporary storage (legacy mode)...");
        long startTime = System.currentTimeMillis();
        
        List<float[]> tempVectors = new ArrayList<>(totalVectorCount.get());
        
        int batchSize = 1000;  // 固定批次大小
        long offset = 0;
        int loadedCount = 0;
        
        try {
            while (offset < totalVectorCount.get()) {
                String sql = "SELECT v.vector_data, v.dimension FROM vectors v ORDER BY v.id LIMIT ? OFFSET ?";
                List<Map<String, Object>> rows = jdbcTemplate.queryForList(sql, batchSize, offset);
                
                for (Map<String, Object> row : rows) {
                    byte[] vectorBytes = (byte[]) row.get("vector_data");
                    int dimension = ((Number) row.get("dimension")).intValue();
                    
                    float[] vector = deserializeVector(vectorBytes, dimension);
                    tempVectors.add(vector);
                    loadedCount++;
                }
                
                offset += rows.size();
                if (loadedCount % 10000 == 0) {
                    log.info("Loaded {}/{} vectors to temp storage", loadedCount, totalVectorCount.get());
                }
            }
        } catch (IOException e) {
            log.error("Failed to deserialize vectors", e);
            throw new RuntimeException("Vector deserialization failed", e);
        }
        
        long duration = System.currentTimeMillis() - startTime;
        log.info("All vectors loaded (legacy): {} vectors in {}ms", tempVectors.size(), duration);
        
        return tempVectors;
    }
    
    /**
     * 内存检查与管理（安全模式核心逻辑）
     * 
     * @param runtime JVM 运行时
     * @param maxMemory 最大堆内存
     * @param loadedCount 已加载向量数
     * @param totalCount 总向量数
     */
    private void checkAndManageMemory(Runtime runtime, long maxMemory, 
                                      int loadedCount, int totalCount) {
        long currentUsed = runtime.totalMemory() - runtime.freeMemory();
        double memoryUsagePercent = (double) currentUsed / maxMemory;
        
        // 情况 1：超过紧急阈值（90%），尝试紧急 GC
        if (memoryUsagePercent > emergencyGcThreshold) {
            log.warn("⚠️ CRITICAL: Memory usage at {}% (threshold: {}%), triggering emergency GC...", 
                String.format("%.1f", memoryUsagePercent * 100),
                String.format("%.1f", emergencyGcThreshold * 100));
            
            // 强制 GC
            System.gc();
            
            try {
                Thread.sleep(200);  // 等待 GC 完成
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
            
            // 检查 GC 效果
            long afterGcUsed = runtime.totalMemory() - runtime.freeMemory();
            double afterGcPercent = (double) afterGcUsed / maxMemory;
            
            if (afterGcPercent > emergencyGcThreshold) {
                // GC 后仍然过高，抛出异常
                String errorMsg = String.format(
                    "Insufficient memory for index building! " +
                    "Current usage: %.1f%% (after GC), Max memory: %dMB, " +
                    "Loaded: %d/%d vectors. " +
                    "Solutions: 1) Increase heap size (-Xmx), 2) Enable safe mode, " +
                    "3) Reduce vector count, 4) Use incremental index building.",
                    afterGcPercent * 100, maxMemory / 1024 / 1024,
                    loadedCount, totalCount
                );
                log.error(errorMsg);
                throw new OutOfMemoryError(errorMsg);
            }
            
            log.info("Emergency GC successful. Memory reduced to {}%", 
                String.format("%.1f", afterGcPercent * 100));
        }
        // 情况 2：超过 GC 阈值（85%），触发常规 GC
        else if (memoryUsagePercent > gcThreshold) {
            log.info("Memory usage at {}% (threshold: {}%), triggering GC...", 
                String.format("%.1f", memoryUsagePercent * 100),
                String.format("%.1f", gcThreshold * 100));
            
            System.gc();
            
            try {
                Thread.sleep(100);  // 短暂等待
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
            
            long afterGcUsed = runtime.totalMemory() - runtime.freeMemory();
            log.info("GC completed. Memory usage: {}%", 
                String.format("%.1f", (double) afterGcUsed / maxMemory * 100));
        }
    }
    
    /**
     * 批量添加向量(不重建索引,由调用方控制何时重建)
     * @param docIds 文档ID列表
     * @param vectors 向量列表
     */
    public synchronized void addVectorsBatch(List<Long> docIds, List<float[]> vectors) {
        if (docIds == null || docIds.isEmpty() || vectors == null || vectors.isEmpty()) {
            log.warn("Empty docIds or vectors list");
            return;
        }
        
        if (docIds.size() != vectors.size()) {
            throw new IllegalArgumentException("docIds and vectors must have the same size");
        }
        
        try {
            // 过滤掉null向量
            List<Long> validDocIds = new ArrayList<>();
            List<float[]> validVectors = new ArrayList<>();
            int skipCount = 0;
            
            for (int i = 0; i < vectors.size(); i++) {
                float[] vector = vectors.get(i);
                if (vector == null) {
                    log.warn("Skipping null vector at index {} for docId {}", i, docIds.get(i));
                    skipCount++;
                    continue;
                }
                validDocIds.add(docIds.get(i));
                validVectors.add(vector);
            }
            
            if (validVectors.isEmpty()) {
                log.error("All vectors are null, cannot add to index");
                return;
            }
            
            if (skipCount > 0) {
                log.warn("Filtered out {} null vectors, remaining: {}", skipCount, validVectors.size());
            }
            
            // 【关键修改】先持久化到数据库，确保数据一致性
            persistVectorsBatch(validDocIds, validVectors);
            
            // 【关键修改】数据库操作成功后，再更新内存状态
            for (int i = 0; i < validVectors.size(); i++) {
                Long docId = validDocIds.get(i);
                float[] vector = validVectors.get(i);
                
                // 放入热缓存
                hotVectorCache.put(docId, vector.clone());
                
                // 分配新的nodeId（原子递增，线程安全）
                int nodeId = totalVectorCount.getAndIncrement();
                nodeIdToDocIdMap.put(nodeId, docId);
                docIdToNodeIdMap.put(docId, nodeId);
            }
            
            // ⚠️ 注意:这里不调用 buildIndex()
            // 由调用方在添加完所有向量后手动调用 rebuildIndex()
            
            log.info("Added {} vectors to database and cache successfully (index not rebuilt yet)", validVectors.size());
            
        } catch (Exception e) {
            log.error("Failed to add vectors batch", e);
            throw new RuntimeException("Add vectors batch failed", e);
        }
    }
    
    /**
     * 批量添加向量 - 仅持久化到数据库（在事务内调用）
     * 
     * 【重要】此方法只负责将向量保存到数据库，不更新内存状态。
     * 内存状态的更新应在事务提交后通过 updateMemoryState() 完成。
     * 
     * 【事务说明】此方法不使用独立事务，而是加入调用方的事务。
     * 这样可以确保 DOCUMENTS 和 VECTORS 表的数据在同一事务中原子性提交或回滚。
     * 
     * @param docIds 文档ID列表
     * @param vectors 向量列表
     */
    public synchronized void persistVectorsOnly(List<Long> docIds, List<float[]> vectors) {
        if (docIds == null || docIds.isEmpty() || vectors == null || vectors.isEmpty()) {
            log.warn("Empty docIds or vectors list");
            return;
        }
        
        if (docIds.size() != vectors.size()) {
            throw new IllegalArgumentException("docIds and vectors must have the same size");
        }
        
        // 过滤掉null向量
        List<Long> validDocIds = new ArrayList<>();
        List<float[]> validVectors = new ArrayList<>();
        int skipCount = 0;
        
        for (int i = 0; i < vectors.size(); i++) {
            float[] vector = vectors.get(i);
            if (vector == null) {
                log.warn("Skipping null vector at index {} for docId {}", i, docIds.get(i));
                skipCount++;
                continue;
            }
            validDocIds.add(docIds.get(i));
            validVectors.add(vector);
        }
        
        if (validVectors.isEmpty()) {
            log.error("All vectors are null, cannot persist");
            return;
        }
        
        if (skipCount > 0) {
            log.warn("Filtered out {} null vectors, remaining: {}", skipCount, validVectors.size());
        }
        
        // 持久化到数据库
        persistVectorsBatch(validDocIds, validVectors);
        
        log.info("Persisted {} vectors to database", validVectors.size());
    }
    
    /**
     * 更新内存状态（在事务提交后调用）
     * 
     * 【重要】此方法必须在事务成功提交后调用，确保数据库和内存状态一致。
     * 建议在 TransactionSynchronization.afterCommit() 中调用。
     * 
     * @param docIds 文档ID列表
     * @param vectors 向量列表
     */
    public synchronized void updateMemoryState(List<Long> docIds, List<float[]> vectors) {
        if (docIds == null || docIds.isEmpty() || vectors == null || vectors.isEmpty()) {
            return;
        }
        
        if (docIds.size() != vectors.size()) {
            throw new IllegalArgumentException("docIds and vectors must have the same size");
        }
        
        int skippedCount = 0;
        for (int i = 0; i < docIds.size(); i++) {
            Long docId = docIds.get(i);
            float[] vector = vectors.get(i);
            
            // 幂等性检查：避免重复添加
            if (docIdToNodeIdMap.containsKey(docId)) {
                log.debug("DocId {} already exists in memory, skipping", docId);
                skippedCount++;
                continue;
            }
            
            // 更新热缓存
            hotVectorCache.put(docId, vector.clone());
            
            // 分配新的nodeId
            int nodeId = totalVectorCount.getAndIncrement();
            nodeIdToDocIdMap.put(nodeId, docId);
            docIdToNodeIdMap.put(docId, nodeId);
        }
        
        log.info("Updated memory state for {} vectors (skipped {} duplicates)", 
            docIds.size() - skippedCount, skippedCount);
    }
    
    /**
     * 异步调度索引重建
     * 
     * 【说明】由于项目已使用 Semaphore 限制单文件上传，
     * 因此不需要防抖机制，直接异步执行即可。
     */
    public void scheduleRebuildIndexAsync() {
        java.util.concurrent.CompletableFuture.runAsync(() -> {
            try {
                log.info("Starting async index rebuild...");
                long startTime = System.currentTimeMillis();
                
                rebuildIndex();
                
                long duration = System.currentTimeMillis() - startTime;
                log.info("Async index rebuild completed successfully in {}ms", duration);
                
            } catch (Exception e) {
                log.error("Async index rebuild failed", e);
                // 不抛出异常，依赖定时任务修复
            }
        });
    }
    
    /**
     * 手动触发索引重建(批量添加后调用)
     * 
     * 【性能策略】
     * 使用 cache-aside 模式：优先从缓存读取，未命中时才查数据库。
     * 因为 addVectorsBatch() 已将新向量放入缓存，所以大部分访问会命中缓存。
     * 相比全量加载，这种方式在增量更新场景下更快。
     */
    public synchronized void rebuildIndex() {
        if (totalVectorCount.get() > 0) {
            long startTime = System.currentTimeMillis();
            log.info("Rebuilding index using cache-aside strategy...");
            
            // 使用 cache-aside 模式（优先从缓存读取）
            buildIndex();
            
            long duration = System.currentTimeMillis() - startTime;
            log.info("Index rebuilt successfully with {} vectors in {}ms", 
                totalVectorCount.get(), duration);
        } else {
            log.warn("No vectors to build index");
        }
    }
    
    public synchronized void addVector(Long docId, float[] vectorArray) {
        try {
            // 【关键修改】先持久化到数据库，确保数据一致性
            persistVector(docId, vectorArray);
            
            // 【关键修改】数据库操作成功后，再更新内存状态
            // 添加到热缓存
            hotVectorCache.put(docId, vectorArray.clone());
            
            // 分配新的nodeId（原子递增，线程安全）
            int nodeId = totalVectorCount.getAndIncrement();
            nodeIdToDocIdMap.put(nodeId, docId);
            docIdToNodeIdMap.put(docId, nodeId);
            
            // 重建索引(单个向量添加时仍需要重建)
            buildIndex();
            
            log.info("Added vector to database and cache successfully for docId={}", docId);
            
        } catch (Exception e) {
            log.error("Failed to add vector for docId={}", docId, e);
            throw new RuntimeException("Add vector failed", e);
        }
    }
    
    /**
     * 将向量持久化到数据库
     */
    private void persistVector(Long docId, float[] vector) {
        try {
            byte[] vectorBytes = serializeVector(vector);
            String sql = "INSERT INTO vectors (doc_id, vector_data, dimension) VALUES (?, ?, ?)";
            jdbcTemplate.update(sql, docId, vectorBytes, vector.length);
        } catch (Exception e) {
            log.error("Failed to persist vector for docId={}", docId, e);
            throw new RuntimeException("Vector persistence failed", e);
        }
    }
    
    /**
     * 批量持久化向量(性能优化)
     * @param docIds 文档ID列表
     * @param vectors 向量列表
     */
    public void persistVectorsBatch(List<Long> docIds, List<float[]> vectors) {
        if (docIds == null || docIds.isEmpty() || vectors == null || vectors.isEmpty()) {
            return;
        }
        
        if (docIds.size() != vectors.size()) {
            throw new IllegalArgumentException("docIds and vectors must have the same size");
        }
        
        try {
            String sql = "INSERT INTO vectors (doc_id, vector_data, dimension) VALUES (?, ?, ?)";
            
            int successCount = 0;
            int skipCount = 0;
            
            // 简化实现:在事务中逐条插入(H2会自动优化)
            for (int i = 0; i < docIds.size(); i++) {
                float[] vector = vectors.get(i);
                
                // 跳过null向量
                if (vector == null) {
                    log.warn("Skipping null vector at index {} for docId {}", i, docIds.get(i));
                    skipCount++;
                    continue;
                }
                
                byte[] vectorBytes = serializeVector(vector);
                jdbcTemplate.update(sql, docIds.get(i), vectorBytes, vector.length);
                successCount++;
            }
            
            if (skipCount > 0) {
                log.warn("Batch persisted {} vectors, skipped {} null vectors", successCount, skipCount);
            } else {
                log.debug("Batch persisted {} vectors", successCount);
            }
        } catch (Exception e) {
            log.error("Failed to batch persist vectors", e);
            throw new RuntimeException("Batch vector persistence failed", e);
        }
    }
    
    /**
     * 真正的批量INSERT(使用JDBC Batch API)
     * 适用于大批量场景(>1000个向量)或远程数据库
     * @param docIds 文档ID列表
     * @param vectors 向量列表
     * @return 成功插入的记录数
     */
    public int persistVectorsBatchWithJdbcBatch(List<Long> docIds, List<float[]> vectors) {
        if (docIds == null || docIds.isEmpty() || vectors == null || vectors.isEmpty()) {
            return 0;
        }
        
        if (docIds.size() != vectors.size()) {
            throw new IllegalArgumentException("docIds and vectors must have the same size");
        }
        
        try {
            String sql = "INSERT INTO vectors (doc_id, vector_data, dimension) VALUES (?, ?, ?)";
            
            // 使用JDBC Batch API进行真正的批量插入
            int[] batchResults = jdbcTemplate.batchUpdate(sql, new org.springframework.jdbc.core.BatchPreparedStatementSetter() {
                @Override
                public void setValues(java.sql.PreparedStatement ps, int i) throws java.sql.SQLException {
                    try {
                        byte[] vectorBytes = serializeVector(vectors.get(i));
                        ps.setLong(1, docIds.get(i));
                        ps.setBytes(2, vectorBytes);
                        ps.setInt(3, vectors.get(i).length);
                    } catch (java.io.IOException e) {
                        throw new java.sql.SQLException("Failed to serialize vector at index " + i, e);
                    }
                }
                
                @Override
                public int getBatchSize() {
                    return docIds.size();
                }
            });
            
            int successCount = 0;
            for (int result : batchResults) {
                if (result > 0) {
                    successCount++;
                }
            }
            
            log.info("JDBC batch persisted {} vectors, success: {}/{}", 
                docIds.size(), successCount, batchResults.length);
            
            return successCount;
            
        } catch (org.springframework.dao.DataAccessException e) {
            log.error("Failed to JDBC batch persist vectors", e);
            
            // 尝试从异常中提取详细信息
            if (e.getCause() instanceof java.sql.BatchUpdateException) {
                java.sql.BatchUpdateException batchEx = (java.sql.BatchUpdateException) e.getCause();
                int[] updateCounts = batchEx.getUpdateCounts();
                log.error("Batch update failed. Completed: {} records, Failed at index: {}",
                    updateCounts.length, 
                    updateCounts.length < docIds.size() ? updateCounts.length : -1);
            }
            
            throw new RuntimeException("JDBC batch vector persistence failed", e);
        }
    }
    
    /**
     * 搜索并返回带分数的结果(用于多查询变体并行检索)
     * @return Map<docId, score>
     */
    public Map<Long, Float> searchWithScores(float[] queryVectorArray, int topK) {
        if (graphIndex == null || graphIndex.size() == 0) {
            log.warn("JVector index is empty");
            return Collections.emptyMap();
        }
        
        try {
            // 转换查询向量
            io.github.jbellis.jvector.vector.types.VectorFloat<?> queryVector = 
                com.myspringairag.util.VectorUtils.toVectorFloat(queryVectorArray);
            
            // 创建RAVV包装器(从缓存/数据库动态获取)
            io.github.jbellis.jvector.graph.RandomAccessVectorValues ravv = 
                new io.github.jbellis.jvector.graph.RandomAccessVectorValues() {
                @Override
                public int size() {
                    return totalVectorCount.get();
                }
                
                @Override
                public int dimension() {
                    return dimensions;
                }
                
                @Override
                public io.github.jbellis.jvector.vector.types.VectorFloat<?> getVector(int i) {
                    Long docId = nodeIdToDocIdMap.get(i);
                    if (docId == null) {
                        throw new IllegalStateException("Invalid nodeId: " + i);
                    }
                    
                    // 先从缓存获取
                    float[] vector = hotVectorCache.getIfPresent(docId);
                    if (vector != null) {
                        return com.myspringairag.util.VectorUtils.toVectorFloat(vector);
                    }
                    
                    // 缓存未命中,从数据库加载
                    vector = loadVectorFromDatabase(docId);
                    if (vector != null) {
                        hotVectorCache.put(docId, vector);
                        return com.myspringairag.util.VectorUtils.toVectorFloat(vector);
                    }
                    
                    throw new IllegalStateException("Vector not found for docId: " + docId);
                }
                
                @Override
                public boolean isValueShared() {
                    return false;
                }
                
                /**
                 * 返回RAVV实例的副本，用于并发搜索。
                 * 
                 * 【安全性说明】
                 * 当前实现直接返回this（自身引用），这是安全的，因为：
                 * 1. 本RAVV是无状态的：只读取外部变量，不维护任何实例级可变状态
                 * 2. nodeIdToDocIdMap：ConcurrentHashMap，线程安全，支持并发读写
                 * 3. hotVectorCache：Caffeine缓存，内部使用ConcurrentHashMap，线程安全
                 * 4. jdbcTemplate：Spring管理的JdbcTemplate，线程安全
                 * 5. loadVectorFromDatabase()：每次调用创建局部变量，无共享状态
                 * 
                 * 所有写操作（cache.put）都委托给线程安全的Caffeine组件。
                 * 因此，多线程共享此RAVV实例是安全的。
                 * 
                 * 【并发场景】
                 * 在ParallelVectorRetrievalService中，多个查询变体可能并行执行搜索，
                 * 每个搜索调用会创建独立的RAVV实例（在方法内部），因此实际上
                 * JVector不会调用此copy()方法。但为了符合接口契约和防御性编程，
                 * 保留此实现并说明其安全性依据。
                 * 
                 * 【注意】
                 * 如果未来修改添加了实例级可变状态（如本地缓存、计数器等），
                 * 此方法必须更新为创建真正的深拷贝实例。
                 */
                @Override
                public io.github.jbellis.jvector.graph.RandomAccessVectorValues copy() {
                    return this;
                }
            };
            
            // 执行向量相似性搜索
            io.github.jbellis.jvector.graph.SearchResult result = 
                io.github.jbellis.jvector.graph.GraphSearcher.search(
                queryVector,
                topK * 2,  // 搜索更多候选
                ravv,
                io.github.jbellis.jvector.vector.VectorSimilarityFunction.COSINE,
                graphIndex,
                io.github.jbellis.jvector.util.Bits.ALL
            );
            
            // 过滤低相似度结果并转换为(docId, score)对
            Map<Long, Float> docScores = Arrays.stream(result.getNodes())
                .filter(r -> r.score >= similarityThreshold)
                .map(r -> new AbstractMap.SimpleEntry<>(
                    nodeIdToDocIdMap.get(r.node), 
                    r.score
                ))
                .filter(entry -> entry.getKey() != null)
                .filter(entry -> !deletedDocIds.contains(entry.getKey()))  // ← 过滤已删除的文档
                .sorted(Map.Entry.<Long, Float>comparingByValue().reversed())
                .limit(topK)
                .collect(Collectors.toMap(
                    Map.Entry::getKey,
                    Map.Entry::getValue,
                    (e1, e2) -> e1,
                    LinkedHashMap::new
                ));
            
            return docScores;
            
        } catch (Exception e) {
            log.error("JVector search with scores failed", e);
            return Collections.emptyMap();
        }
    }
    
    /**
     * 删除指定文件的所有向量(软删除策略)
     * 
     * 【性能优化】
     * 采用软删除策略：只标记删除，不立即重建索引。
     * 搜索时会自动过滤已删除的文档，保证结果准确性。
     * 索引会在定时任务中定期清理重建（默认每天凌晨2点）。
     * 
     * 【重要】删除后会立即触发索引清理，确保映射关系一致。
     * 
     * @param filename 文件名
     * @param docIds 要删除的文档ID集合
     */
    public synchronized void removeVectorsForFilename(String filename, Set<Long> docIds) {
        try {
            // 第1步：标记为已删除（添加到软删除集合）
            deletedDocIds.addAll(docIds);
            log.info("Marked {} vectors as deleted for filename: {}", docIds.size(), filename);
            
            // 第2步：从热缓存中移除（避免搜索时返回已删除的）
            for (Long docId : docIds) {
                hotVectorCache.invalidate(docId);
            }
            
            // 第3步：从数据库中批量删除（性能优化）
            if (!docIds.isEmpty()) {
                batchRemoveFromDatabase(docIds);
            }
            
            log.info("Soft-deleted {} vectors. Rebuilding index immediately...", docIds.size());
            
            // 第4步：立即清理并重建索引（确保映射关系一致）
            forceCleanupDeletedVectors();
            
        } catch (Exception e) {
            log.error("Failed to remove vectors for filename={}", filename, e);
            throw new RuntimeException("Remove vectors failed", e);
        }
    }
    
    /**
     * 从数据库中删除向量
     */
    private void removeFromDatabase(Long docId) {
        try {
            String sql = "DELETE FROM vectors WHERE doc_id = ?";
            jdbcTemplate.update(sql, docId);
        } catch (Exception e) {
            log.error("Failed to remove vector from database for docId={}", docId, e);
        }
    }
    
    /**
     * 批量从数据库中删除向量（性能优化）
     * @param docIds 要删除的文档ID列表
     */
    private void batchRemoveFromDatabase(Set<Long> docIds) {
        if (docIds == null || docIds.isEmpty()) {
            return;
        }
        
        try {
            // 使用IN子句进行批量删除
            String placeholders = docIds.stream()
                .map(String::valueOf)
                .collect(Collectors.joining(","));
            String sql = "DELETE FROM vectors WHERE doc_id IN (" + placeholders + ")";
            
            int deletedCount = jdbcTemplate.update(sql);
            log.info("Batch deleted {} vectors from database", deletedCount);
        } catch (Exception e) {
            log.error("Failed to batch remove vectors from database for {} docIds", docIds.size(), e);
            // 降级处理：如果批量删除失败，回退到逐条删除
            log.warn("Falling back to individual deletion due to batch failure");
            for (Long docId : docIds) {
                removeFromDatabase(docId);
            }
        }
    }
    
    /**
     * 强制立即清理已删除向量（同步执行，阻塞调用方）
     * 用于deletedDocIds超过上限时的紧急清理
     */
    private void forceCleanupDeletedVectors() {
        log.info("=== Forcing immediate cleanup of {} deleted vectors ===", deletedDocIds.size());
        long startTime = System.currentTimeMillis();
        
        try {
            // 注意：即时清理不扫描孤儿数据，只处理已知被标记删除的docId
            performCleanupWithoutOrphanDetection();
            
            long duration = System.currentTimeMillis() - startTime;
            log.info("=== Forced cleanup completed in {}ms ===", duration);
        } catch (Exception e) {
            log.error("Forced cleanup failed", e);
            // 不抛出异常，允许继续删除操作
        }
    }
    
    /**
     * 从数据库加载单个向量(用于缓存未命中时)
     */
    private float[] loadVectorFromDatabase(Long docId) {
        try {
            String sql = "SELECT v.vector_data, v.dimension FROM vectors v WHERE v.doc_id = ? LIMIT 1";
            List<Map<String, Object>> rows = jdbcTemplate.queryForList(sql, docId);
            
            if (rows.isEmpty()) {
                log.warn("Vector not found in database for docId={}", docId);
                return null;
            }
            
            Map<String, Object> row = rows.get(0);
            byte[] vectorBytes = (byte[]) row.get("vector_data");
            int dimension = ((Number) row.get("dimension")).intValue();
            
            return deserializeVector(vectorBytes, dimension);
        } catch (Exception e) {
            log.error("Failed to load vector from database for docId={}", docId, e);
            return null;
        }
    }
    
    /**
     * 序列化向量为字节数组
     */
    private byte[] serializeVector(float[] vector) throws IOException {
        if (vector == null) {
            throw new IllegalArgumentException("Vector cannot be null");
        }
        
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        DataOutputStream dos = new DataOutputStream(baos);
        
        for (float v : vector) {
            dos.writeFloat(v);
        }
        
        dos.close();
        return baos.toByteArray();
    }
    
    /**
     * 从字节数组反序列化向量
     */
    private float[] deserializeVector(byte[] bytes, int dimension) throws IOException {
        ByteArrayInputStream bais = new ByteArrayInputStream(bytes);
        DataInputStream dis = new DataInputStream(bais);
        
        float[] vector = new float[dimension];
        for (int i = 0; i < dimension; i++) {
            vector[i] = dis.readFloat();
        }
        
        dis.close();
        return vector;
    }
    
    public int getIndexSize() {
        return graphIndex != null ? graphIndex.size() : 0;
    }
    
    /**
     * 获取热缓存统计信息
     */
    public String getCacheStats() {
        var stats = hotVectorCache.stats();
        return String.format(
            "Cache Stats - Size: %d, Hit Rate: %.2f%%, Hits: %d, Misses: %d, Evictions: %d",
            hotVectorCache.estimatedSize(),
            stats.hitRate() * 100,
            stats.hitCount(),
            stats.missCount(),
            stats.evictionCount()
        );
    }
    
    /**
     * 内存监控:定时检查内存使用情况
     */
    @Scheduled(fixedRate = 60000)  // 每分钟检查一次
    public void monitorMemoryUsage() {
        Runtime runtime = Runtime.getRuntime();
        long usedMemory = runtime.totalMemory() - runtime.freeMemory();
        long maxMemory = runtime.maxMemory();
        double usagePercent = (double) usedMemory / maxMemory * 100;
        
        // 格式化百分比（SLF4J不支持{:.1f}语法）
        String formattedPercent = String.format("%.1f", usagePercent);
        
        log.info("Memory Monitor - Used: {}MB / Max: {}MB ({}%), Vectors: {}, Cache: {}",
            usedMemory / 1024 / 1024,
            maxMemory / 1024 / 1024,
            formattedPercent,
            totalVectorCount.get(),
            hotVectorCache.estimatedSize());
        
        // 监控待清理的软删除向量数量
        int pendingCleanupCount = deletedDocIds.size();
        if (pendingCleanupCount > 0) {
            log.info("Pending cleanup: {} vectors marked as deleted (will be cleaned immediately or by scheduled task)", 
                pendingCleanupCount);
        }
        
        // 告警阈值:80%
        if (usagePercent > 80) {
            log.warn("⚠️ HIGH MEMORY USAGE: {}% - Consider increasing heap size or reducing cache size",
                formattedPercent);
        }
        
        // 输出缓存统计
        log.debug(getCacheStats());
    }
    
    /**
     * 定时清理已删除向量并重建索引
     * 
     * 【执行时间】每天凌晨2点
     * 【目的】清理软删除的向量 + 扫描孤儿数据，释放索引空间，优化搜索性能
     * 【注意】此操作会触发全量索引重建，建议在低峰期执行
     */
    @Scheduled(cron = "0 0 2 * * ?")  // 每天凌晨2点执行
    public void cleanupDeletedVectors() {
        log.info("=== Starting scheduled cleanup task ===");
        
        // 使用独立线程执行，添加超时保护
        ExecutorService executor = Executors.newSingleThreadExecutor();
        Future<?> future = executor.submit(() -> {
            performCleanupWithOrphanDetection();
        });
        
        try {
            // 最多等待30分钟
            future.get(30, TimeUnit.MINUTES);
            log.info("Cleanup task completed successfully");
        } catch (TimeoutException e) {
            log.error("⚠️ Cleanup task timed out after 30 minutes, cancelling...");
            future.cancel(true);  // 中断任务
        } catch (Exception e) {
            log.error("Cleanup task failed", e);
        } finally {
            executor.shutdown();
        }
    }
    
    /**
     * 执行实际的清理逻辑（在独立线程中运行）
     * 包含孤儿数据检测和清理
     */
    private void performCleanupWithOrphanDetection() {
        long startTime = System.currentTimeMillis();
        
        try {
            // 第0步：扫描孤儿数据（vectors表中存在但documents表中不存在的doc_id）
            log.info("Step 0: Scanning for orphaned vectors...");
            Set<Long> orphanedDocIds = scanOrphanedVectors();
            if (!orphanedDocIds.isEmpty()) {
                log.warn("Found {} orphaned vectors in database", orphanedDocIds.size());
                // 将孤儿数据加入软删除集合
                deletedDocIds.addAll(orphanedDocIds);
                // 从数据库中批量删除孤儿向量（性能优化）
                if (!orphanedDocIds.isEmpty()) {
                    batchRemoveFromDatabase(orphanedDocIds);
                }
                log.info("Removed {} orphaned vectors from database", orphanedDocIds.size());
            } else {
                log.info("No orphaned vectors found");
            }
            
            // 第1-4步：执行通用清理逻辑
            performCleanupCore(orphanedDocIds.size());
            
        } catch (Exception e) {
            log.error("Failed to cleanup deleted vectors", e);
            // 不清空deletedDocIds，下次重试
            throw new RuntimeException("Cleanup failed", e);
        }
    }
    
    /**
     * 执行清理逻辑但不扫描孤儿数据（用于即时清理）
     */
    private void performCleanupWithoutOrphanDetection() {
        long startTime = System.currentTimeMillis();
        
        try {
            // 直接执行通用清理逻辑（孤儿数据数量为0）
            performCleanupCore(0);
            
        } catch (Exception e) {
            log.error("Failed to cleanup deleted vectors", e);
            // 不清空deletedDocIds，下次重试
            throw new RuntimeException("Cleanup failed", e);
        }
    }
    
    /**
     * 核心清理逻辑：重建索引并清空软删除集合
     * @param orphanCount 孤儿数据数量（用于日志记录）
     */
    private void performCleanupCore(int orphanCount) {
        // 第1步：收集要保留的向量映射
        Map<Integer, Long> newNodeIdMap = new HashMap<>();
        Map<Long, Integer> newDocIdMap = new HashMap<>();
        
        int newNodeIdCount = 0;
        for (Map.Entry<Integer, Long> entry : nodeIdToDocIdMap.entrySet()) {
            Long docId = entry.getValue();
            if (!deletedDocIds.contains(docId)) {
                newNodeIdMap.put(newNodeIdCount, docId);
                newDocIdMap.put(docId, newNodeIdCount);
                newNodeIdCount++;
            }
        }
        
        int removedCount = nodeIdToDocIdMap.size() - newNodeIdCount;
        log.info("Rebuilding index with {} vectors (removed {})", newNodeIdCount, removedCount);
        
        // 第2步：更新状态
        nodeIdToDocIdMap = newNodeIdMap;
        docIdToNodeIdMap = newDocIdMap;
        totalVectorCount.set(newNodeIdCount);
        
        // 第3步：使用临时存储重建索引（同启动时的优化）
        // 【注意】此操作可能需要较长时间，取决于向量数量和内存配置
        if (totalVectorCount.get() > 0) {
            log.info("Starting full vector reload for index rebuild (this may take a while)...");
            List<float[]> tempVectors = loadAllVectorsTemporarily();
            buildIndexWithTempStorage(tempVectors);
            tempVectors.clear();
            tempVectors = null;
        } else {
            graphIndex = null;
        }
        
        // 第4步：清空软删除集合
        deletedDocIds.clear();
        
        if (orphanCount > 0) {
            log.info("=== Cleanup completed. Removed {} vectors (including {} orphans) ===", 
                removedCount, orphanCount);
        } else {
            log.info("=== Cleanup completed. Removed {} vectors ===", removedCount);
        }
    }
    
    /**
     * 扫描孤儿数据：查找vectors表中存在但documents表中不存在的doc_id
     * @return 孤儿docId集合
     */
    private Set<Long> scanOrphanedVectors() {
        Set<Long> orphanedDocIds = new HashSet<>();
        
        try {
            // 查询vectors表中所有唯一的doc_id
            String sql = "SELECT DISTINCT v.doc_id FROM vectors v " +
                        "LEFT JOIN documents d ON v.doc_id = d.id " +
                        "WHERE d.id IS NULL";
            
            List<Long> results = jdbcTemplate.queryForList(sql, Long.class);
            orphanedDocIds.addAll(results);
            
            if (!orphanedDocIds.isEmpty()) {
                log.warn("Detected {} orphaned vectors (exist in vectors table but not in documents table)", 
                    orphanedDocIds.size());
            }
            
        } catch (Exception e) {
            log.error("Failed to scan orphaned vectors", e);
        }
        
        return orphanedDocIds;
    }
    
    /**
     * 获取软删除统计信息（用于监控）
     * @return 包含deletedDocIds数量、总向量数、删除比例等信息
     */
    public Map<String, Object> getDeletedStats() {
        Map<String, Object> stats = new HashMap<>();
        stats.put("deletedDocIdsCount", deletedDocIds.size());
        stats.put("totalVectorCount", totalVectorCount.get());
        stats.put("deletionRatio", totalVectorCount.get() > 0 ? 
            (double) deletedDocIds.size() / totalVectorCount.get() : 0.0);
        stats.put("cleanupStrategy", "immediate_after_delete_with_scheduled_backup");
        return stats;
    }
    
    /**
     * 手动触发清理（供管理员调用）
     * 可通过Controller暴露此接口，允许管理员手动清理已删除向量
     */
    public synchronized void manualCleanup() {
        log.info("Manual cleanup triggered by admin");
        cleanupDeletedVectors();
    }
}
