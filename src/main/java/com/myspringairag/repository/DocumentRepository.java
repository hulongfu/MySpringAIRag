package com.myspringairag.repository;

import com.myspringairag.model.Document;
import com.myspringairag.service.QueryTokenizationService;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.jdbc.core.RowMapper;
import org.springframework.stereotype.Repository;

import java.sql.ResultSet;
import java.sql.SQLException;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;

@Slf4j
@Repository
public class DocumentRepository {
    
    private final JdbcTemplate jdbcTemplate;
    
    @Autowired
    private QueryTokenizationService tokenizationService;
    
    private static final RowMapper<Document> ROW_MAPPER = new RowMapper<Document>() {
        @Override
        public Document mapRow(ResultSet rs, int rowNum) throws SQLException {
            Document doc = new Document();
            doc.setId(rs.getLong("id"));
            doc.setFilename(rs.getString("filename"));
            doc.setContent(rs.getString("content"));
            doc.setParentContent(rs.getString("parent_content"));  // 添加parentContent
            doc.setChunkIndex(rs.getInt("chunk_index"));
            doc.setTotalChunks(rs.getInt("total_chunks"));
            doc.setUploadTime(rs.getTimestamp("upload_time").toLocalDateTime());
            doc.setFileSize(rs.getLong("file_size"));
            doc.setMimeType(rs.getString("mime_type"));
            return doc;
        }
    };
    
    public DocumentRepository(JdbcTemplate jdbcTemplate) {
        this.jdbcTemplate = jdbcTemplate;
    }
    
    public Document save(Document document) {
        String sql = "INSERT INTO documents (filename, content, parent_content, chunk_index, total_chunks, file_size, mime_type) " +
                    "VALUES (?, ?, ?, ?, ?, ?, ?)";
        
        jdbcTemplate.update(sql, 
            document.getFilename(),
            document.getContent(),
            document.getParentContent(),  // 添加parentContent
            document.getChunkIndex(),
            document.getTotalChunks(),
            document.getFileSize(),
            document.getMimeType()
        );
        
        // 获取最后插入的ID
        Long id = jdbcTemplate.queryForObject("SELECT MAX(id) FROM documents", Long.class);
        document.setId(id);
        return document;
    }
    
    public Optional<Document> findById(Long id) {
        String sql = "SELECT * FROM documents WHERE id = ?";
        List<Document> results = jdbcTemplate.query(sql, ROW_MAPPER, id);
        return results.isEmpty() ? Optional.empty() : Optional.of(results.get(0));
    }
    
    public List<Document> findByIds(List<Long> ids) {
        if (ids == null || ids.isEmpty()) {
            return List.of();
        }
        
        String placeholders = String.join(",", ids.stream().map(id -> "?").toArray(String[]::new));
        String sql = "SELECT * FROM documents WHERE id IN (" + placeholders + ")";
        
        return jdbcTemplate.query(sql, ROW_MAPPER, ids.toArray());
    }
    
    public List<Document> findByFilename(String filename) {
        String sql = "SELECT * FROM documents WHERE filename = ? ORDER BY chunk_index";
        return jdbcTemplate.query(sql, ROW_MAPPER, filename);
    }
    
    public List<String> getAllFilenames() {
        String sql = "SELECT filename FROM documents GROUP BY filename ORDER BY MAX(upload_time) DESC";
        return jdbcTemplate.queryForList(sql, String.class);
    }
    
    public void deleteByFilename(String filename) {
        String sql = "DELETE FROM documents WHERE filename = ?";
        jdbcTemplate.update(sql, filename);
    }
    
    public void deleteById(Long id) {
        String sql = "DELETE FROM documents WHERE id = ?";
        jdbcTemplate.update(sql, id);
    }
    
    public List<Document> findAll() {
        String sql = "SELECT * FROM documents ORDER BY upload_time DESC, chunk_index ASC";
        return jdbcTemplate.query(sql, ROW_MAPPER);
    }
    
    public long count() {
        return jdbcTemplate.queryForObject("SELECT COUNT(*) FROM documents", Long.class);
    }
    
    /**
     * 关键词搜索（使用IK分词器 + LIKE模糊匹配）
     * @param query 查询文本
     * @param limit 返回结果数量限制
     * @return 文档ID列表
     */
    public List<Long> keywordSearch(String query, int limit) {
        // 使用IK分词器提取关键词
        String[] keywords = tokenizationService.tokenize(query);
        
        if (keywords.length == 0) {
            return List.of();
        }
        
        // 构建LIKE查询
        StringBuilder sqlBuilder = new StringBuilder();
        sqlBuilder.append("SELECT DISTINCT id FROM documents WHERE ");
        
        for (int i = 0; i < keywords.length; i++) {
            if (i > 0) {
                sqlBuilder.append(" OR ");
            }
            sqlBuilder.append("LOWER(content) LIKE LOWER(?)");
        }
        
        sqlBuilder.append(" LIMIT ?");
        
        // 准备参数（添加通配符）
        Object[] params = new Object[keywords.length + 1];
        for (int i = 0; i < keywords.length; i++) {
            params[i] = "%" + keywords[i] + "%";
        }
        params[keywords.length] = limit;
        
        try {
            List<Long> results = jdbcTemplate.queryForList(sqlBuilder.toString(), Long.class, params);
            return results;
        } catch (Exception e) {
            log.error("Keyword search failed: {}", e.getMessage());
            return List.of();
        }
    }
}
