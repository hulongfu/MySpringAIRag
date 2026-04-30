package com.myspringairag.service;

import com.myspringairag.model.TextChunk;
import lombok.extern.slf4j.Slf4j;
import org.apache.tika.Tika;
import org.apache.tika.exception.TikaException;
import org.apache.tika.metadata.Metadata;
import org.apache.tika.parser.ParseContext;
import org.apache.tika.parser.Parser;
import org.apache.tika.parser.microsoft.OfficeParser;
import org.apache.tika.parser.pdf.PDFParser;
import org.apache.tika.sax.BodyContentHandler;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;
import org.xml.sax.ContentHandler;
import org.xml.sax.SAXException;

import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

@Slf4j
@Service
public class DocumentParserService {
    
    @Value("${app.upload-dir}")
    private String uploadDir;
    
    @Value("${app.chunk-size}")
    private int chunkSize;
    
    @Value("${app.chunk-overlap}")
    private int chunkOverlap;
    
    @Value("${app.use-semantic-chunking:false}")
    private boolean useSemanticChunking;
    
    @Value("${app.semantic-similarity-threshold:0.7}")
    private float semanticSimilarityThreshold;
    
    @Autowired
    private SemanticTextSplitter semanticSplitter;
    
    @Autowired
    private HierarchicalChunkingService hierarchicalChunkingService;
    
    private final Tika tika = new Tika();
    
    public String parseFile(MultipartFile file) throws IOException, TikaException, SAXException {
        // 保存文件
        Path uploadPath = Paths.get(uploadDir);
        Files.createDirectories(uploadPath);
        
        Path filePath = uploadPath.resolve(file.getOriginalFilename());
        Files.copy(file.getInputStream(), filePath, StandardCopyOption.REPLACE_EXISTING);
        
        // 解析文件内容
        return extractText(filePath);
    }
    
    /**
     * 从文件路径解析文本（用于异步处理）
     */
    public String parseFileFromPath(Path filePath) throws IOException, TikaException, SAXException {
        return extractText(filePath);
    }
    
    public String extractText(Path filePath) throws IOException, TikaException, SAXException {
        String filename = filePath.getFileName().toString().toLowerCase();
        
        // 对于.txt和.md文件，直接使用UTF-8读取
        if (filename.endsWith(".txt") || filename.endsWith(".md")) {
            return new String(Files.readAllBytes(filePath), java.nio.charset.StandardCharsets.UTF_8);
        }
        
        // 其他文件类型使用Tika解析
        Parser parser = getParserForFile(filePath);
        
        // 如果没有特定解析器，使用Tika自动检测
        if (parser == null) {
            return tika.parseToString(filePath.toFile());
        }
        
        Metadata metadata = new Metadata();
        // 强制设置UTF-8编码
        metadata.set(Metadata.CONTENT_ENCODING, "UTF-8");
        ContentHandler handler = new BodyContentHandler(-1); // 无限制
        
        try (InputStream stream = Files.newInputStream(filePath)) {
            parser.parse(stream, handler, metadata, new ParseContext());
        }
        
        String text = handler.toString();
        return text;
    }
    
    private Parser getParserForFile(Path filePath) {
        String filename = filePath.getFileName().toString().toLowerCase();
        
        if (filename.endsWith(".pdf")) {
            return new PDFParser();
        } else if (filename.endsWith(".txt") || filename.endsWith(".md")) {
            return null; // 使用默认解析器
        }
        
        // 对于 Office 文档（.doc, .docx, .xls, .xlsx, .ppt, .pptx 等）
        // 返回 null，让 Tika 自动检测并选择正确的解析器
        // 这样可以正确处理 Office 2003 (.doc) 和 Office 2007+ (.docx) 格式
        return null;
    }
    
    public List<String> splitIntoChunks(String text) {
        if (text == null || text.isEmpty()) {
            return new ArrayList<>();
        }
        
        log.info("Text length: {} characters", text.length());
        
        String docId = "doc_" + System.currentTimeMillis();
        List<TextChunk> chunks;
        
        if (useSemanticChunking) {
            // 使用层级分块（新方案）
            log.info("Using hierarchical chunking (Parent-Child strategy)");
            // 注意：这里暂时保留原有逻辑，后续可以完全切换到hierarchicalChunkingService
            chunks = semanticSplitter.semanticChunk(
                docId, 
                text, 
                semanticSimilarityThreshold,
                chunkSize,
                20  // max sentences per chunk
            );
        } else {
            // 使用固定大小分块
            log.info("Using fixed-size chunking: size={}, overlap={}", chunkSize, chunkOverlap);
            chunks = semanticSplitter.splitBySentences(docId, text, chunkSize, chunkOverlap);
        }
        
        // 转换为 String 列表（保持与原有接口兼容）
        List<String> chunkTexts = chunks.stream()
            .map(TextChunk::getContent)
            .collect(Collectors.toList());
        
        log.info("Split text into {} chunks", chunkTexts.size());
        return chunkTexts;
    }


}
