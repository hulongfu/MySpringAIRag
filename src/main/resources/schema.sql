CREATE TABLE IF NOT EXISTS documents (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    filename VARCHAR(255) NOT NULL,
    content TEXT NOT NULL,  -- 小块（200 tokens，用于向量检索）
    parent_content TEXT,  -- 大块（语义完整，用于提供上下文）
    chunk_index INT NOT NULL,
    total_chunks INT NOT NULL,
    upload_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    file_size BIGINT,
    mime_type VARCHAR(100)
);

CREATE TABLE IF NOT EXISTS vectors (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    doc_id BIGINT NOT NULL,
    vector_data BLOB NOT NULL,
    dimension INT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (doc_id) REFERENCES documents(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_filename ON documents(filename);
CREATE INDEX IF NOT EXISTS idx_upload_time ON documents(upload_time);
CREATE INDEX IF NOT EXISTS idx_doc_id ON vectors(doc_id);
