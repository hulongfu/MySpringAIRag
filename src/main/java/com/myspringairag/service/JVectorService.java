package com.myspringairag.service;

import io.github.jbellis.jvector.graph.*;
import io.github.jbellis.jvector.util.Bits;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import jakarta.annotation.PostConstruct;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.stereotype.Service;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;
import java.util.stream.Collectors;

import static com.myspringairag.util.VectorUtils.toVectorFloat;

/**
 * JVector向量服务 - 基于JVector 3.0.6 + H2数据库持久化
 * 使用HNSW算法进行高效向量搜索
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
    
    // JVector索引
    private GraphIndex graphIndex;
    
    // 映射：nodeId -> docId
    private Map<Integer, Long> nodeIdToDocIdMap = new HashMap<>();
    
    // 内存中存储所有原始向量（用于重建RAVV）
    private List<float[]> allVectors = new ArrayList<>();
    
    public JVectorService(JdbcTemplate jdbcTemplate) {
        this.jdbcTemplate = jdbcTemplate;
    }
    
    @PostConstruct
    public void init() {
        log.info("Initializing JVector index with dimensions={}", dimensions);
        
        try {
            // 创建索引目录
            Path indexDir = Paths.get(indexPath);
            Files.createDirectories(indexDir);
            
            // 从数据库加载或构建索引
            loadOrBuildIndex();
            
            log.info("JVector index initialized successfully with {} vectors", 
                     graphIndex != null ? graphIndex.size() : 0);
            
        } catch (Exception e) {
            log.error("Failed to initialize JVector index", e);
            throw new RuntimeException("JVector initialization failed", e);
        }
    }
    
    /**
     * 从数据库加载数据并构建索引
     */
    private void loadOrBuildIndex() throws IOException {
        log.info("Loading vectors from database...");
        
        // 检查vectors表是否存在
        String checkTableSql = "SELECT COUNT(*) FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_NAME = 'VECTORS'";
        Integer tableCount = jdbcTemplate.queryForObject(checkTableSql, Integer.class);
        
        if (tableCount == null || tableCount == 0) {
            log.info("Vectors table does not exist yet, skipping index build");
            return;
        }
        
        // 从数据库加载所有向量
        List<float[]> loadedVectors = new ArrayList<>();
        Map<Integer, Long> tempNodeIdMap = new HashMap<>();
        
        String sql = "SELECT v.id, v.doc_id, v.vector_data, v.dimension FROM vectors v ORDER BY v.id";
        List<Map<String, Object>> rows = jdbcTemplate.queryForList(sql);
        
        int idx = 0;
        for (Map<String, Object> row : rows) {
            Long docId = ((Number) row.get("doc_id")).longValue();
            byte[] vectorBytes = (byte[]) row.get("vector_data");
            int dimension = ((Number) row.get("dimension")).intValue();
            
            // 反序列化向量
            float[] vectorArray = deserializeVector(vectorBytes, dimension);
            
            loadedVectors.add(vectorArray);
            tempNodeIdMap.put(idx++, docId);
        }
        
        if (loadedVectors.isEmpty()) {
            log.info("No vectors found in database");
            return;
        }
        
        // 更新状态
        this.allVectors = loadedVectors;
        this.nodeIdToDocIdMap = tempNodeIdMap;
        
        // 构建索引
        buildIndex();
        
        // 保存索引到文件
        saveIndex();
        
        log.info("Successfully built JVector index with {} vectors", graphIndex.size());
    }
    
    /**
     * 构建索引
     */
    private void buildIndex() {
        // 创建RandomAccessVectorValues包装器
        RandomAccessVectorValues ravv = new RandomAccessVectorValues() {
            @Override
            public int size() {
                return allVectors.size();
            }
            
            @Override
            public int dimension() {
                return dimensions;
            }
            
            @Override
            public VectorFloat<?> getVector(int i) {
                return toVectorFloat(allVectors.get(i));
            }
            
            @Override
            public boolean isValueShared() {
                return false;
            }
            
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
        
        // 获取索引（build会自动从RAVV读取所有向量）
        this.graphIndex = builder.build(ravv);
    }
    
    /**
     * 保存索引到文件（当前实现为从数据库重建，无需文件持久化）
     */
    private void saveIndex() throws IOException {
        // 索引在应用启动时从数据库重建，无需额外文件持久化
    }
    
    /**
     * 批量添加向量（不重建索引，由调用方控制何时重建）
     * @param docIds 文档ID列表
     * @param vectors 向量列表
     */
    public synchronized void addVectorsBatch(List<Long> docIds, List<float[]> vectors) {
        if (docIds == null || docIds.isEmpty() || vectors == null || vectors.isEmpty()) {
            return;
        }
        
        if (docIds.size() != vectors.size()) {
            throw new IllegalArgumentException("docIds and vectors must have the same size");
        }
        
        try {
            // 添加到内存列表
            for (int i = 0; i < vectors.size(); i++) {
                allVectors.add(vectors.get(i));
                int nodeId = allVectors.size() - 1;
                nodeIdToDocIdMap.put(nodeId, docIds.get(i));
            }
            
            // 批量持久化到数据库
            persistVectorsBatch(docIds, vectors);
            
            // ⚠️ 注意：这里不调用 buildIndex()
            // 由调用方在添加完所有向量后手动调用 rebuildIndex()
            
            log.info("Added {} vectors to memory (index not rebuilt yet)", vectors.size());
            
        } catch (Exception e) {
            log.error("Failed to add vectors batch", e);
            throw new RuntimeException("Add vectors batch failed", e);
        }
    }
    
    /**
     * 手动触发索引重建（批量添加后调用）
     */
    public synchronized void rebuildIndex() {
        if (!allVectors.isEmpty()) {
            long startTime = System.currentTimeMillis();
            buildIndex();
            long duration = System.currentTimeMillis() - startTime;
            log.info("Index rebuilt successfully with {} vectors in {}ms", 
                allVectors.size(), duration);
        } else {
            log.warn("No vectors to build index");
        }
    }
    
    public synchronized void addVector(Long docId, float[] vectorArray) {
        try {
            // 添加到内存列表
            allVectors.add(vectorArray);
            int nodeId = allVectors.size() - 1;
            nodeIdToDocIdMap.put(nodeId, docId);
            
            // 持久化到数据库
            persistVector(docId, vectorArray);
            
            // 重建索引（单个向量添加时仍需要重建）
            buildIndex();
            
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
     * 批量持久化向量（性能优化）
     * @param docIds 文档ID列表
     * @param vectors 向量列表
     */
    public void persistVectorsBatch(List<Long> docIds, List<float[]> vectors) {
        if (docIds == null || docIds.isEmpty() || vectors == null || vectors.isEmpty()) {
            return;
        }
        
        try {
            String sql = "INSERT INTO vectors (doc_id, vector_data, dimension) VALUES (?, ?, ?)";
            
            // 简化实现：在事务中逐条插入（H2会自动优化）
            for (int i = 0; i < docIds.size(); i++) {
                byte[] vectorBytes = serializeVector(vectors.get(i));
                jdbcTemplate.update(sql, docIds.get(i), vectorBytes, vectors.get(i).length);
            }
            
            log.debug("Batch persisted {} vectors", docIds.size());
        } catch (Exception e) {
            log.error("Failed to batch persist vectors", e);
            throw new RuntimeException("Batch vector persistence failed", e);
        }
    }
    
    /**
     * 真正的批量INSERT（使用JDBC Batch API）
     * 适用于大批量场景（>1000个向量）或远程数据库
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
    
    public synchronized List<Long> search(float[] queryVectorArray, int topK) {
        try {
            if (graphIndex == null || graphIndex.size() == 0) {
                return Collections.emptyList();
            }
            
            // 转换查询向量为VectorFloat
            VectorFloat<?> queryVector = toVectorFloat(queryVectorArray);
            
            // 创建RandomAccessVectorValues用于搜索
            RandomAccessVectorValues ravv = new RandomAccessVectorValues() {
                @Override
                public int size() {
                    return allVectors.size();
                }
                
                @Override
                public int dimension() {
                    return dimensions;
                }
                
                @Override
                public VectorFloat<?> getVector(int i) {
                    return toVectorFloat(allVectors.get(i));
                }
                
                @Override
                public boolean isValueShared() {
                    return false;
                }
                
                @Override
                public RandomAccessVectorValues copy() {
                    return this;
                }
            };
            
            // 执行向量相似性搜索
            SearchResult result = GraphSearcher.search(
                queryVector,
                topK * 2,  // 搜索更多候选
                ravv,
                VectorSimilarityFunction.COSINE,
                graphIndex,
                Bits.ALL
            );
            
            // 过滤低相似度结果并转换为docId
            List<Long> docIds = Arrays.stream(result.getNodes())
                .filter(r -> r.score >= similarityThreshold)
                .map(r -> nodeIdToDocIdMap.get(r.node))
                .filter(Objects::nonNull)
                .limit(topK)
                .collect(Collectors.toList());
            
            return docIds;
            
        } catch (Exception e) {
            log.error("JVector search failed", e);
            return Collections.emptyList();
        }
    }
    
    /**
     * 搜索并返回带分数的结果（用于多查询变体并行检索）
     * @return Map<docId, score>
     */
    public synchronized Map<Long, Float> searchWithScores(float[] queryVectorArray, int topK) {
        if (graphIndex == null || graphIndex.size() == 0) {
            log.warn("JVector index is empty");
            return Collections.emptyMap();
        }
        
        try {
            // 转换查询向量
            io.github.jbellis.jvector.vector.types.VectorFloat<?> queryVector = 
                com.myspringairag.util.VectorUtils.toVectorFloat(queryVectorArray);
            
            // 创建RAVV包装器
            io.github.jbellis.jvector.graph.RandomAccessVectorValues ravv = 
                new io.github.jbellis.jvector.graph.RandomAccessVectorValues() {
                @Override
                public int size() {
                    return allVectors.size();
                }
                
                @Override
                public int dimension() {
                    return dimensions;
                }
                
                @Override
                public io.github.jbellis.jvector.vector.types.VectorFloat<?> getVector(int i) {
                    return com.myspringairag.util.VectorUtils.toVectorFloat(allVectors.get(i));
                }
                
                @Override
                public boolean isValueShared() {
                    return false;
                }
                
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
    
    public synchronized void removeVectorsForFilename(String filename, Set<Long> docIds) {
        try {
            // 收集要保留的向量
            List<float[]> retainedVectors = new ArrayList<>();
            Map<Integer, Long> newNodeIdMap = new HashMap<>();
            
            int newNodeId = 0;
            for (int i = 0; i < allVectors.size(); i++) {
                Long docId = nodeIdToDocIdMap.get(i);
                if (docId != null && !docIds.contains(docId)) {
                    retainedVectors.add(allVectors.get(i));
                    newNodeIdMap.put(newNodeId, docId);
                    newNodeId++;
                }
            }
            
            // 更新状态
            allVectors = retainedVectors;
            nodeIdToDocIdMap = newNodeIdMap;
            
            // 重建索引
            if (!allVectors.isEmpty()) {
                buildIndex();
            }
            
            // 从数据库中删除
            for (Long docId : docIds) {
                removeFromDatabase(docId);
            }
            
            log.info("Removed {} vectors, remaining: {}", docIds.size(), allVectors.size());
            
        } catch (Exception e) {
            log.error("Failed to remove vectors", e);
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
     * 序列化向量为字节数组
     */
    private byte[] serializeVector(float[] vector) throws IOException {
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
}
