# MySpringAIRag - 智能RAG知识库问答系统

基于 **JVector + H2 + Spring AI** 的轻量级 RAG（检索增强生成）应用，采用层级分块策略、并行多查询变体检索和关键词分数调整，显著提升检索准确性。

## 🎯 核心特性

✅ **层级分块（Parent-Child Chunking）**  
- 小块（~200 tokens）用于向量检索，保证embedding质量
- 大块（语义完整）用于提供上下文给LLM
- 按行拆分，不切断代码、命令等关键信息
- 小块之间重叠20 tokens，避免信息断裂

✅ **语义分块优化** ⭐ NEW
- 相似度阈值提升至 0.85，确保语义一致性
- 最小分块保护：每块至少包含3个句子
- 避免单句成块，提升分块质量
- 自适应技术文档特点

✅ **并行多查询变体检索**  
- 根据查询长度动态生成变体（<10字: 3个，10-20字: 5个，>20字: 6个）
- 变体类型：原始查询、重写查询、核心实体提取、简化版、关键词扩展
- 并行执行向量检索
- RRF（Reciprocal Rank Fusion）融合多路结果
- 显著提升召回率

✅ **关键词分数调整** ⭐ NEW
- 基于IK分词器提取核心关键词
- 根据关键词匹配比例调整文档相似度分数
- 公式：新分数 = 原分数 × (匹配关键词数 / 总关键词数)
- 提升检索结果的相关性

✅ **本地Embedding模型**  
- 使用 BGE-Small-ZH-v1.5 ONNX模型（384维）
- 无需API调用，完全离线运行
- 支持中文语义理解

✅ **云端Chat模型**  
- 集成硅基流动（SiliconFlow）DeepSeek-V3.2
- 高质量答案生成

✅ **智能文档解析**  
- 支持 TXT、PDF、DOC、DOCX、MD 格式
- Apache Tika 自动识别文档类型并选择正确的解析器
- 完美兼容 Office 2003 (.doc) 和 Office 2007+ (.docx) 格式

✅ **SSE 实时进度推送** ⭐ NEW
- Server-Sent Events 实时推送处理进度（10%, 30%, 60%, 90%, 100%）
- 前端先建立 SSE 连接，再生成 taskId 并发起上传
- 避免小文件处理太快错过进度消息
- 自动管理 SSE 连接生命周期，防止内存泄漏
- 智能区分正常关闭和异常断开，避免误报错误

✅ **并发控制与临时文件管理** ⭐ NEW
- Semaphore 信号量限制同时只能有一个文件在处理
- 临时文件保存在项目根目录 `uploads/` 文件夹
- 处理完成后自动清理临时文件（finally 块保证）
- 防止重复提交和状态混乱

✅ **前后端一体化**  
- Thymeleaf + 原生 HTML/CSS/JavaScript
- 简洁易用的Web界面

## 🏗️ 技术架构

### 技术栈

- **后端框架**: Spring Boot 3.4.3 + Spring AI 1.0.0-M6
- **向量搜索**: JVector 3.0.6（纯Java实现，HNSW算法）
- **数据库**: H2 2.3.232（嵌入式数据库）
- **Embedding模型**: BGE-Small-ZH-v1.5 ONNX（本地，384维）
- **Chat模型**: SiliconFlow DeepSeek-V3.2（云端API）
- **文档解析**: Apache Tika 2.9.1
- **中文分词**: IKAnalyzer 2012_u6
- **实时通信**: Server-Sent Events (SSE)
- **并发控制**: Java Semaphore
- **前端**: Thymeleaf + 原生 HTML/CSS/JavaScript

### 系统架构图

```
┌─────────────┐
│  用户提问    │
└──────┬──────┘
       ↓
┌─────────────────────┐
│  QueryRewriteService │ ← 查询重写（保留关键词）
└──────┬──────────────┘
       ↓
┌──────────────────────────┐
│ ParallelVectorRetrieval   │ ← 根据查询长度生成变体
│ Service                   │    (<10字:3个, 10-20字:5个, >20字:6个)
│                           │
│  ├─ Variant 1: 原始查询   │
│  ├─ Variant 2: 查询重写   │
│  ├─ Variant 3: 核心实体   │
│  ├─ Variant 4: 简化版     │
│  └─ Variant 5+: 关键词扩展│
└──────┬───────────────────┘
       ↓ (并行执行)
┌─────────────────────┐
│ EmbeddingService     │ ← 本地BGE模型生成向量
└──────┬──────────────┘
       ↓
┌─────────────────────┐
│ JVectorService       │ ← HNSW向量检索
│                      │
│  Top-K per variant   │
└──────┬──────────────┘
       ↓
┌─────────────────────┐
│ RRF Fusion           │ ← 倒数排名融合
│ (Reciprocal Rank     │
│  Fusion)             │
└──────┬──────────────┘
       ↓
┌─────────────────────┐
│ DocumentRepository   │ ← 从H2查询Document对象
│                      │    (content + parentContent)
└──────┬──────────────┘
       ↓
┌─────────────────────┐
│ RagService           │ ← 构建上下文
│ buildContext()       │    优先返回parentContent
└──────┬──────────────┘
       ↓
┌─────────────────────┐
│ ChatClient           │ ← 调用SiliconFlow API
│ (DeepSeek-V3.2)      │
└──────┬──────────────┘
       ↓
┌─────────────┐
│  返回答案    │
└─────────────┘
```

### 数据流图（上传文档 - SSE 异步流程）⭐

```
┌─────────────┐
│  用户选择文件 │
└──────┬──────┘
       ↓
┌─────────────────────┐
│  前端生成 UUID       │ ← generateUUID()
│  (taskId)            │
└──────┬──────────────┘
       ↓
┌─────────────────────┐
│  前端建立 SSE 连接    │ ← EventSource(/api/knowledge/stream/{taskId})
│  监听进度             │   自动关闭旧连接，防止内存泄漏
└──────┬──────────────┘
       ↓
┌─────────────────────┐
│  前端发起上传请求     │ ← POST /api/upload (file + taskId)
└──────┬──────────────┘
       ↓
┌─────────────────────┐
│  RagController       │ ← 检查文件大小 (<50MB)
│                      │   获取信号量许可 (Semaphore.tryAcquire)
└──────┬──────────────┘
       ↓
┌─────────────────────┐
│ AsyncUploadService   │ ← 保存临时文件到 uploads/
│ .submitUploadWith    │   temp_{timestamp}_{filename}
│  TaskId()            │
└──────┬──────────────┘
       ↓ (@Async 异步执行)
┌─────────────────────┐
│ processDocument()    │ ← 在独立线程中处理
│                      │
│  1. notifyProgress   │ → SSE: 10% "正在读取文件..."
│     (10%)            │
│  2. notifyProgress   │ → SSE: 30% "正在解析文档..."
│     (30%)            │
│  3. ragService.      │
│     uploadDocument   │ ← 解析、分块、向量化、存储
│  4. notifyProgress   │ → SSE: 90% "正在生成向量索引..."
│     (90%)            │
│  5. notifyCompletion │ → SSE: 100% "文档处理完成！"
│     ()               │
└──────┬──────────────┘
       ↓ (finally 块)
┌─────────────────────┐
│  清理临时文件         │ ← Files.deleteIfExists(tempFile)
│  释放信号量           │ ← uploadSemaphore.release()
└─────────────────────┘
       ↓
┌─────────────────────┐
│  前端收到 COMPLETED  │ ← 清空 selectedFile 和 file-input
│  事件，重置 UI 状态   │   恢复按钮为"上传文档"
└─────────────────────┘
```

## 📁 项目结构

```
MySpringAIRag/
├── src/main/java/com/myspringairag/
│   ├── configuration/
│   │   ├── ChatConfiguration.java        # Chat模型配置
│   │   └── RerankerConfiguration.java    # 重排序配置（可选）
│   ├── controller/
│   │   ├── HomeController.java           # Web页面控制器
│   │   ├── RagController.java            # RAG API控制器
│   │   └── SseController.java            # SSE 进度推送控制器 ⭐
│   ├── model/
│   │   ├── Document.java                 # 文档模型（含parentContent）
│   │   └── ScoredDocument.java           # 带分数的文档
│   ├── repository/
│   │   └── DocumentRepository.java       # H2数据库操作
│   ├── service/
│   │   ├── AsyncUploadService.java       # 异步上传服务 ⭐
│   │   ├── DocumentParserService.java    # 文档解析服务
│   │   ├── EmbeddingService.java         # 本地Embedding服务
│   │   ├── HierarchicalChunkingService.java # 层级分块服务 ⭐
│   │   ├── JVectorService.java           # JVector向量索引服务
│   │   ├── ParallelVectorRetrievalService.java # 并行检索服务 ⭐
│   │   ├── QueryRewriteService.java      # 查询重写服务
│   │   ├── QueryTokenizationService.java # IK分词服务
│   │   ├── RagService.java               # RAG核心服务
│   │   └── SemanticTextSplitter.java     # 语义分块器
│   └── MySpringAIRagApplication.java     # 启动类
├── src/main/resources/
│   ├── templates/
│   │   └── index.html                    # Web界面（含SSE逻辑）
│   ├── application.yml                   # 配置文件
│   └── schema.sql                        # 数据库表结构
├── data/                                 # H2数据库文件
│   ├── ragdb.mv.db
│   └── ragdb.trace.db
├── uploads/                              # 临时文件存储目录 ⭐
│   └── (处理完成后自动删除)
└── pom.xml
```

## 🚀 快速开始

### 1. 环境要求

- **JDK**: 17+（推荐 JDK 17.0.8）
- **Maven**: 3.6+
- **内存**: 建议 4GB+（本地Embedding模型需要加载）

### 2. 配置说明

编辑 `src/main/resources/application.yml`：

```yaml
spring:
  ai:
    # 云端Chat模型配置（硅基流动）
    openai:
      base-url: https://api.siliconflow.cn/v1
      api-key: YOUR_API_KEY  # 替换为你的API Key
      chat:
        options:
          model: deepseek-ai/DeepSeek-V3.2
    
    # 本地Embedding模型配置
    transformers:
      embedding:
        model-path: D:/ideaSpace/MyPython/models/bge-small-zh-v1.5-ONNX
        dimension: 384

app:
  # 分块配置（层级分块）
  chunk-size: 500           # 固定分块时使用（已弃用）
  chunk-overlap: 100        # 固定分块时使用（已弃用）
  use-semantic-chunking: true  # ✅ 启用层级分块
  semantic-similarity-threshold: 0.85  # ✅ 语义相似度阈值（提高至0.85）
  
  # 检索配置
  top-k: 5                  # 最终返回的文档数量
  use-reranking: false      # ❌ 禁用重排序（节省性能）
  rerank-top-k: 50          # 重排序候选数量（已禁用）
  
  # 并行检索配置
  use-parallel-retrieval: true  # ✅ 启用并行多查询变体检索
  
  # 文件上传配置
  upload-dir: D:/tmp/MySpringAIRag/uploads  # 临时文件存储路径

jvector:
  dimensions: 384           # 向量维度（与BGE-Small-ZH匹配）
  top-k: 50                 # 向量检索返回的候选数量
  similarity-threshold: 0.0 # 相似度阈值（0=不过滤）

# Spring Boot 文件上传限制
spring:
  servlet:
    multipart:
      max-file-size: 50MB      # 单个文件最大50MB
      max-request-size: 50MB   # 请求总大小最大50MB
```

### 3. 构建项目

```bash
cd D:\tmp\MySpringAIRag
mvn clean package
```

### 4. 运行应用

```bash
mvn spring-boot:run
```

或者运行打包后的JAR：

```bash
java -jar target/MySpringAIRag-1.0.0.jar
```

### 5. 访问应用

打开浏览器访问：**http://localhost:8080**

H2 控制台（调试用）：http://localhost:8080/h2-console
- JDBC URL: `jdbc:h2:file:./data/ragdb`
- 用户名: `sa`
- 密码: (留空)

## 📖 使用说明

### 上传文档（SSE 异步流程）⭐

1. 点击左侧"点击或拖拽文件到此处上传"区域
2. 选择要上传的文档（支持 TXT、PDF、DOC、DOCX、MD，最大50MB）
3. 点击"上传文档"按钮
4. 系统会自动：
   - **前端生成 UUID** 作为 taskId
   - **建立 SSE 连接** 监听进度（`/api/knowledge/stream/{taskId}`）
   - **发起上传请求** 携带文件和 taskId
   - **后端保存临时文件** 到 `uploads/` 目录
   - **异步处理文档**（解析 → 层级分块 → 向量化 → 存储）
   - **实时推送进度**（10% → 30% → 90% → 100%）
   - **自动清理临时文件**（处理完成后立即删除）
   - **释放信号量**（允许下一个文件上传）
5. 上传完成后，前端自动清空文件选择，恢复按钮状态

**并发控制**：
- 同一时间只能有一个文件在处理（Semaphore 信号量）
- 如果已有文件在处理，新请求会收到 429 错误提示
- 防止 H2 数据库并发冲突和资源竞争

### 智能问答

1. 在右侧输入框中输入问题
2. 按 Enter 或点击"发送"按钮
3. 系统会：
   - **查询重写**（保留关键词）
   - **生成5个查询变体**
   - **并行向量检索**
   - **RRF融合结果**
   - 返回大块上下文给LLM
   - 生成答案

### 管理文档

- 查看左侧"已上传文档"列表
- 点击"删除"按钮可删除指定文档及其所有chunks

## 🔧 核心组件详解

### 1. HierarchicalChunkingService（层级分块服务）⭐

**职责**：实现Parent-Child Chunking策略

**工作流程**：
1. **语义分块**：使用SemanticTextSplitter生成大块（无大小限制）
2. **二次切分**：如果大块 > 200 tokens，按行拆分成小块
3. **行完整性保护**：不切断同一行（除非行本身 > 300 tokens）
4. **重叠处理**：相邻小块之间保留20 tokens重叠

**关键参数**：
- `SMALL_CHUNK_TOKENS = 200`：小块大小
- `OVERLAP_TOKENS = 20`：重叠大小
- `LONG_LINE_THRESHOLD = 300`：超长行阈值
- `MIN_SENTENCES_PER_CHUNK = 3`：最小分块句子数 ⭐ NEW
- `SIMILARITY_THRESHOLD = 0.85`：语义相似度阈值 ⭐ UPDATED

**示例**：
```
原文档：
"""
启动Open WebUI有两种方式：
1. Docker: docker run ...
2. 命令行: open-webui serve
"""

层级分块后：
小块1: "启动Open WebUI有两种方式："
  → parentContent: "启动Open WebUI有两种方式：\n1. Docker: docker run ...\n2. 命令行: open-webui serve"

小块2: "1. Docker: docker run ..."
  → parentContent: 同上

小块3: "2. 命令行: open-webui serve"
  → parentContent: 同上
```

### 2. ParallelVectorRetrievalService（并行检索服务）⭐

**职责**：生成多个查询变体，并行执行向量检索

**工作流程**：
1. **查询重写**：QueryRewriteService优化查询
2. **生成变体**：根据查询长度动态创建变体
   - 简单查询（<10字）：3个变体
   - 中等查询（10-20字）：5个变体
   - 复杂查询（>20字）：6个变体
   - 变体类型：原始查询、重写查询、核心实体、简化版、关键词扩展
3. **并行检索**：线程池并行执行多次向量检索
4. **RRF融合**：倒数排名融合多路结果

**RRF算法**：
```
RRF_Score(doc) = Σ (1 / (K + rank_i))
其中 K=60, rank_i是doc在第i路检索中的排名
```

### 3. QueryTokenizationService（关键词提取服务）⭐ NEW

**职责**：使用IK分词器提取查询的核心关键词

**工作流程**：
1. **中文分词**：使用IKAnalyzer进行智能分词
2. **停用词过滤**：去除常见停用词
3. **核心关键词提取**：保留名词、动词等实词
4. **去重处理**：移除重复关键词

**应用场景**：
- 用于RagService中的关键词分数调整
- 提升检索结果的相关性

### 4. EmbeddingService（本地Embedding服务）

**职责**：使用本地BGE模型生成向量

**特点**：
- 模型：BGE-Small-ZH-v1.5 ONNX
- 维度：384
- 优势：无需API调用，完全离线
- 劣势：首次加载较慢（约5秒）

### 5. JVectorService（向量索引服务）

**职责**：管理JVector向量索引

**特点**：
- 算法：HNSW（Hierarchical Navigable Small World）
- 相似度：余弦相似度
- 持久化：重启时从H2重建索引
- 性能：支持百万级向量检索

### 6. AsyncUploadService（异步上传服务）⭐ NEW

**职责**：管理文件上传的异步处理和进度推送

**工作流程**：
1. **接收文件**：从控制器接收 MultipartFile 和 taskId
2. **保存临时文件**：保存到 `uploads/temp_{timestamp}_{filename}`
3. **提交异步任务**：使用 `@Async` 在独立线程中处理
4. **进度推送**：通过 SseController 实时推送进度
5. **清理资源**：finally 块中删除临时文件并释放信号量

**关键特性**：
- `Semaphore` 并发控制：只允许1个并发任务
- `@Async("documentProcessingExecutor")`：单线程池避免 H2 并发冲突
- 临时文件命名：使用时间戳避免重名
- 自动清理：无论成功或失败都会删除临时文件

### 7. SseController（SSE 进度推送控制器）⭐ NEW

**职责**：管理 SSE 连接和进度推送

**关键方法**：
- `streamProgress(taskId)`：创建 SSE 连接
- `notifyProgress(taskId, progress, status)`：推送进度消息
- `notifyCompletion(taskId, message)`：推送完成消息

**前端管理**：
- 全局变量 `currentEventSource` 管理当前连接
- 建立新连接前自动关闭旧连接，防止内存泄漏
- 收到 COMPLETED 事件后清空文件选择状态

### 8. RagService（RAG核心服务）⭐ UPDATED

**职责**：协调整个RAG流程

**关键方法**：
- `uploadDocument()`：上传文档并建立索引
- `answerQuestion()`：回答问题
  - 查询重写
  - 并行多查询变体检索
  - RRF融合
  - **关键词分数调整** ⭐ NEW
- `buildContext()`：构建上下文（优先返回parentContent）
- `adjustScoreByKeywords()`：根据关键词匹配比例调整分数 ⭐ NEW

## ⚙️ 配置优化指南

### 调整分块策略

```yaml
app:
  use-semantic-chunking: true  # 启用层级分块
  semantic-similarity-threshold: 0.85  # ✅ 提高阈值，确保语义一致性（最新）
```

**语义分块优化说明**：
- **相似度阈值 0.85**：只有语义非常相近的句子才会被聚合到同一块
- **最小分块保护**：每块至少包含3个句子，避免单句成块
- **适用场景**：技术文档、API文档等需要完整上下文的场景

### 调整检索精度

```yaml
jvector:
  top-k: 50              # 增加候选数量→提高召回率
  similarity-threshold: 0.0  # 降低阈值→更多候选

app:
  top-k: 5               # 最终返回数量
  use-parallel-retrieval: true  # 启用并行检索
  parallel-query-variants: 5    # 默认变体数量（实际根据查询长度动态调整）
```

### 性能 vs 精度权衡

| 配置 | 精度高 | 速度快 |
|------|--------|--------|
| jvector.top-k | 50-100 | 10-20 |
| parallel-query-variants | 5-7 | 2-3 |
| use-reranking | true | false |
| semantic-similarity-threshold | 0.8-0.9 | 0.6-0.7 |
| min-sentences-per-chunk | 3-5 | 1-2 |

## ⚠️ 注意事项

### 1. 数据库表结构更新

如果添加了新字段（如`parent_content`），需要手动执行SQL：

```sql
ALTER TABLE documents ADD COLUMN parent_content TEXT;
```

H2数据库不会自动更新已有表的结构。

### 2. 内存占用

- **本地Embedding模型**：约500MB
- **JVector索引**：每个向量约2KB（384维float）
- **建议内存**：4GB+（1000个文档约需1GB）

### 3. 并发限制

- **H2 不适合高并发场景**
- **适合企业内部小团队使用**（<50并发用户）
- **生产环境建议迁移到 PostgreSQL + pgvector**
- **文件上传并发控制**：Semaphore 限制同时只能有1个文件在处理

### 4. API配额

- SiliconFlow有免费额度（每月100万tokens）
- 注意监控API使用情况
- 可切换到其他提供商（OpenAI、Anthropic等）

### 5. 索引重建

- 重启应用时会自动从H2重建JVector索引
- 大量文档时重建较慢（1000个文档约10秒）
- 可通过日志查看进度：`Successfully built JVector index with XXX vectors`

### 6. 临时文件管理 ⭐ NEW

- 临时文件保存在 `uploads/` 目录（可配置 `app.upload-dir`）
- 处理完成后**立即删除**（finally 块保证）
- 如果应用异常退出，可能需要手动清理 `uploads/` 目录
- 建议使用绝对路径配置，避免工作目录问题

### 7. SSE 连接管理 ⭐ NEW

- 前端使用全局变量 `currentEventSource` 管理连接
- 建立新连接前会自动关闭旧连接
- 如果浏览器长时间不刷新，可能会积累多个 EventSource 对象
- 建议在上传完成后刷新页面，或者确保 SSE 连接正确关闭

## 🐛 常见问题

### Q1: 为什么检索不到相关内容？

**可能原因**：
1. 文档未正确上传（检查日志是否有错误）
2. 查询表述不准确（尝试不同问法）
3. 相似度阈值过高（降低`jvector.similarity-threshold`）
4. 分块过大导致embedding稀释（启用层级分块）
5. **关键词匹配度低** ⭐ NEW：查询中的核心关键词未在文档中出现

**解决方案**：
- 查看日志中的`=== Retrieved Context ===`部分
- 检查是否召回了相关chunks
- 调整`jvector.top-k`和`similarity-threshold`
- **优化查询表述**，确保包含关键术语

### Q2: 为什么分块内容太短？⭐ UPDATED

**可能原因**：
1. 语义相似度阈值过低（旧版本默认0.65）
2. 缺少最小分块保护
3. 技术文档中相邻句子语义差异较大

**解决方案**：
- ✅ **已优化**：相似度阈值提升至 0.85
- ✅ **已优化**：添加最小分块保护（至少3个句子）
- 如果仍有问题，可进一步调整：
  ```yaml
  semantic-similarity-threshold: 0.9  # 更严格的语义一致性
  ```

### Q3: 为什么parent_content为空？

**可能原因**：
1. 未启用层级分块（`use-semantic-chunking: false`）
2. DocumentRepository未读取parent_content字段

**解决方案**：
- 确认配置文件中`use-semantic-chunking: true`
- 检查`DocumentRepository.ROW_MAPPER`是否包含`doc.setParentContent(rs.getString("parent_content"))`

### Q4: 应用启动很慢？

**可能原因**：
1. 本地Embedding模型加载慢（首次约5秒）
2. JVector索引重建慢（大量文档时）

**解决方案**：
- 耐心等待首次加载
- 后续启动会快很多
- 减少文档数量或优化分块策略

### Q5: 如何查看数据库中存储的内容？

使用H2控制台：
```sql
-- 查看所有文档
SELECT id, filename, chunk_index, 
       LENGTH(content) as content_length,
       LENGTH(parent_content) as parent_content_length
FROM documents 
LIMIT 10;

-- 查看特定文档的所有chunks
SELECT * FROM documents 
WHERE filename = 'AI应用开发-无名.txt'
ORDER BY chunk_index;
```

### Q6: 为什么上传同一个文件会被重复处理？

**可能原因**：
1. 前端状态未正确清空（selectedFile 仍保留）
2. SSE 连接未正确关闭，导致多次触发
3. 用户在上传过程中多次点击按钮

**解决方案**：
- 确保后端返回 success 后立即清空 selectedFile
- SSE 收到 COMPLETED 事件后再次清空文件选择
- 使用 Semaphore 防止并发处理同一文件
- 刷新浏览器可以彻底清除所有状态

### Q7: uploads 目录下有很多临时文件怎么办？

**正常情况**：
- 临时文件应该在处理完成后立即删除
- 如果应用正常运行，uploads 目录应该是空的

**异常情况**：
- 如果应用异常退出（kill -9、断电等），临时文件可能未被删除
- 可以手动删除 uploads 目录下的所有 `temp_*` 文件
- 这些文件不影响已上传的文档（已存入 H2 数据库）

### Q8: SSE 连接一直不关闭怎么办？

**排查方法**：
1. 打开浏览器开发者工具 → Network → EventStream
2. 检查是否有多个 SSE 连接处于 Pending 状态
3. 查看控制台是否有 SSE error 日志

**解决方案**：
- 刷新浏览器页面
- 检查前端代码是否正确管理 currentEventSource
- 确保后端在 COMPLETED 事件中调用了 emitter.complete()

## 🔮 未来改进方向

- [ ] 索引持久化（避免重启时重建）
- [ ] 增量更新（只处理新增/修改的文档）
- [ ] 支持更多文档格式（Excel、PPT、图片OCR）
- [ ] 文档元数据过滤（按文件名、上传时间等筛选）
- [ ] 向量缓存机制（相同查询不重复计算）
- [ ] 多轮对话上下文（记住历史对话）
- [ ] 用户认证和权限管理
- [ ] 迁移到 PostgreSQL + pgvector（生产环境）
- [ ] 支持分布式部署
- [x] ✅ 语义分块优化（相似度阈值 + 最小分块保护）⭐ NEW
- [x] ✅ 关键词分数调整（IK分词 + 匹配比例）⭐ NEW

## 📄 许可证

MIT License

## 📞 联系方式

如有问题或建议，欢迎反馈！

---

**最后更新**: 2026-04-30  
**版本**: 1.3.0  
**主要更新**:
- ✅ 新增 SSE 实时进度推送功能（基于真实业务节点）
- ✅ 新增并发控制（Semaphore）
- ✅ 优化临时文件管理（自动清理）
- ✅ 修复前端重复提交问题
- ✅ 完善 SSE 连接管理（避免误报“连接中断”）
- ✅ 修复 Word 文档解析问题（支持 .doc 和 .docx）
- ✅ 优化进度推送逻辑（移除伪延迟，基于实际业务节点）
- ✅ **语义分块优化**：相似度阈值提升至 0.85，添加最小分块保护 ⭐ NEW
- ✅ **关键词分数调整**：基于IK分词提取核心关键词，按匹配比例调整分数 ⭐ NEW
- ✅ **删除冗余代码**：移除未使用的RRF融合方法和调试日志
