# 长三角/京津粤人工智能政策分析 Agent 项目目标文档

## 1. 项目背景

本项目的目标不是做一个泛泛的“聊天机器人”或简单的知识库问答 Demo，而是围绕 **求职导向的 Agent 工程项目** 来设计，目标岗位偏向：

- Agent 开发工程师
- AI 应用工程师
- RAG / LLM 应用开发工程师
- 大模型应用架构师（Agent 方向）
- 偏工程落地的智能体开发岗位

项目设计需要兼顾以下几个现实约束：

1. **要能做完**：不能依赖难以获取的大规模私有数据，也不能过度依赖复杂集群环境。
2. **要能讲清楚**：项目结构、模块边界、工程取舍、评测方式要明确，适合面试表达。
3. **要贴近 JD**：覆盖 Agent、RAG、工具调用、Memory、评测、工程化这些招聘高频要求。
4. **要体现成长轨迹**：结合近期学习内容，从数据清洗、chunk、embedding、retrieval、memory、tool calling、LangGraph/工作流，到评测和服务化，形成一个完整项目。
5. **要适合个人开发**：优先选择公开政策数据，避免在数据采集阶段卡死。

---

## 2. 项目定位

### 2.1 项目名称

**Policy Agent：长三角 / 京津粤人工智能政策分析智能体**

### 2.2 一句话描述

构建一个面向公开政策文档的 **RAG + Agent 系统**，支持多地区人工智能政策的文档接入、清洗、结构化切分、检索问答、政策比较、政策摘要与企业画像匹配，并具备基础评测和工程化能力。

### 2.3 项目核心价值

该项目主要解决以下问题：

- 政策文档分散、冗长，人工阅读成本高
- 不同地区政策口径不同，横向比较困难
- 企业或个人很难快速判断某项政策是否适配自己
- 普通 RAG 只能“问答”，无法体现更强的任务分解、工具调用和结构化输出能力

因此项目应当从“**文档问答**”升级为“**任务型政策分析 Agent**”。

---

## 3. 目标用户与典型场景

### 3.1 目标用户

- 求职项目面试官（项目展示对象）
- AI / 创业 / 产业研究方向的政策关注者
- 中小企业 / 创业团队的政策查询用户
- 需要做地区政策比较的人

### 3.2 典型使用场景

1. **政策检索问答**
   - 例如：“上海近两年有哪些人工智能支持政策？”
   - 例如：“哪些政策提到算力券或模型券？”

2. **政策摘要**
   - 例如：“总结这份政策的支持重点、适用对象和申报条件。”

3. **政策比较**
   - 例如：“对比上海和深圳在人工智能支持政策上的差异。”
   - 例如：“北京和苏州在大模型支持方面有哪些不同？”

4. **政策匹配**
   - 例如：“我们是一家做工业视觉的小型 AI 企业，注册在苏州，有哪些值得关注的政策？”
   - 例如：“一家初创大模型应用公司，适合重点关注哪些城市支持条款？”

---

## 4. 项目整体目标

### 4.1 总体目标

完成一个可运行、可演示、可评测的智能体项目，覆盖：

- 多源文档接入（PDF / HTML / TXT）
- 文本清洗与标准化
- 标题感知的 chunk 切分
- 向量检索与引用返回
- Agent 工作流路由
- 至少 3~4 个工具调用能力
- 基础 Memory 能力
- 基础评测体系
- 命令行或简单 API 演示

### 4.2 求职导向目标

该项目最终应能够在简历和面试中证明以下能力：

1. **Python 工程能力**
2. **RAG 基础链路能力**
   - ingest
   - clean
   - chunk
   - embedding
   - retrieval
   - rerank（可后补）
3. **Agent 工作流能力**
   - 路由
   - 节点编排
   - 工具调用
4. **Memory / 状态管理能力**
5. **评测意识**
   - retrieval 指标
   - answer 可用率
   - latency
6. **工程化意识**
   - 模块拆分
   - 配置化
   - 日志
   - 可重复执行 pipeline

---

## 5. 本项目要体现的知识图谱

结合近期学习与求职目标，本项目应综合体现以下知识点：

### 5.1 Python 工程基础

- 包结构设计
- dataclass / Pydantic 数据模型
- 脚本入口
- 配置管理
- 文件处理
- 日志与错误处理

### 5.2 RAG 基础

- metadata 设计
- chunking
- embedding
- vector store
- retrieval
- top-k
- 证据引用

### 5.3 文档工程

- 多源文件加载（pdf/html/txt）
- 清洗规则设计
- 标题识别
- chunk 结构保持

### 5.4 Agent 能力

- intent routing
- workflow / graph 思路
- tool calling
- state 管理
- 可扩展到 LangGraph

### 5.5 Memory

- 短期上下文
- 偏好保存
- 会话摘要（后续）

### 5.6 评测与工程化

- retrieval recall
- answer quality
- 失败样本分析
- script 化 pipeline
- metadata + raw + clean + chunks 可追溯

### 5.7 求职表达点

项目需要能让面试官看到：

- 不只是“会调 API”
- 不只是“会用 LangChain”
- 而是理解完整数据链路、工作流、文档处理、评测与工程取舍

---

## 6. 项目功能范围

### 6.1 V1 必做功能

1. **政策文档接入**
   - 从 metadata 读取文档信息
   - 支持 PDF / HTML / TXT 三类源文件

2. **文档清洗**
   - 去掉噪声文本
   - 保留正文标题
   - 统一 clean txt 格式

3. **标题感知的分块**
   - 保留“一、（一）1.” 等层级标题
   - 优先按标题切，再按长度切

4. **向量检索**
   - 根据 query 检索相关政策 chunk
   - 返回文本和来源信息

5. **基础问答**
   - 给定问题，结合检索结果生成回答
   - 附带引用来源

6. **基础政策摘要**
   - 对单篇政策输出结构化摘要

### 6.2 V2 建议功能

1. **政策比较**
   - 比较两个地区或两份政策的异同

2. **企业画像匹配**
   - 根据企业画像从候选政策中匹配

3. **更完整的 Agent 工作流**
   - route -> retrieve -> tool -> answer

4. **会话 Memory**
   - 保留用户偏好：地区、主题、输出风格

5. **离线评测**
   - retrieval recall
   - answer 可用率
   - chunk 质量抽查

### 6.3 暂不优先功能

以下功能可以后补，不作为第一阶段目标：

- 多 Agent 协作
- 完整 Web 前端
- K8s / Docker 生产部署
- 微调 / SFT / DPO
- 大规模自动爬取系统
- 复杂可视化报表

---

## 7. 技术栈建议

### 7.1 编程语言

- **Python 3.11+**

原因：
- Agent / RAG 生态成熟
- 便于快速开发和脚本处理
- 贴近目标岗位要求

### 7.2 核心框架与库

#### 文档处理
- `PyMuPDF` 或 `pdfplumber`：PDF 文本提取
- `BeautifulSoup4` / `lxml`：HTML 正文提取
- 标准库 `pathlib`、`re`、`csv`、`json`

#### 数据模型
- `pydantic` 或 `dataclasses`
- 推荐 V1 可先用 `dataclasses`，轻量简单

#### RAG 检索
- embedding 模型接口（可后接 OpenAI / 本地 embedding）
- 向量存储：
  - V1 推荐：`FAISS`
  - 或轻量本地方案
- rerank：V1 可预留接口，先不强依赖

#### Agent / 工作流
- V1 可以先手写 workflow
- V2 可升级接入 **LangGraph**
- 这样既能体现底层理解，也便于后续演进

#### API / 演示
- `FastAPI`（V2 可接入）
- V1 先用脚本和 CLI demo 也可以

#### 测试
- `pytest`

#### 配置
- `PyYAML`
- `.env` + `python-dotenv`

#### 日志
- Python `logging`

---

## 8. 项目目录结构建议

```text
policy-agent/
├── app/
│   ├── ingest/
│   ├── clean/
│   ├── chunk/
│   ├── retrieval/
│   ├── tools/
│   ├── agent/
│   ├── models/
│   ├── utils/
│   └── main.py
├── configs/
├── data/
│   ├── raw/
│   │   ├── anhui/
│   │   ├── jiangsu/
│   │   ├── zhejiang/
│   │   ├── shanghai/
│   │   ├── beijing/
│   │   ├── shenzhen/
│   │   └── guangdong/
│   ├── clean/
│   ├── chunks/
│   ├── metadata/
│   └── vector_store/
├── scripts/
├── tests/
├── requirements.txt
└── README.md
```

### 目录组织原则

- 按 **地区优先** 管理原始文件，贴合政策比较场景
- 处理流程按 **raw -> clean -> chunks -> vector_store** 组织
- 代码结构按功能分层，不混杂

---

## 9. 数据组织方案

### 9.1 原始文件组织

按地区 + 文件类型组织，例如：

```text
data/raw/jiangsu/html/JS002.html
data/raw/zhejiang/pdf/ZJ001.pdf
data/raw/beijing/html/BJ001.html
```

### 9.2 清洗后文件组织

```text
data/clean/jiangsu/JS002.txt
data/clean/zhejiang/ZJ001.txt
```

### 9.3 chunk 文件组织

```text
data/chunks/jiangsu/JS002.jsonl
data/chunks/zhejiang/ZJ001.jsonl
```

### 9.4 metadata 设计原则

metadata 不仅用于展示信息，更是后续：

- 构建路径
- 检索过滤
- 回答引用
- 比较分析

的基础。

建议至少包含：

- doc_id
- title
- region
- level
- issuer
- publish_date
- policy_type
- theme
- tier
- status
- source_format
- doc_no
- source_url
- notes

---

## 10. 数据清洗设计原则

### 10.1 核心原则

- **原始文件不动**
- **清洗脚本可重复执行**
- **规则清洗为主，人工抽查为辅**
- **大模型只处理难例，不做主力清洗**

### 10.2 为什么不主要靠大模型清洗

1. 成本高
2. 速度慢
3. 容易改写原文
4. 难复现
5. 不利于工程稳定性

### 10.3 清洗后的目标格式

清洗后的 clean txt 不是简单正文，而是一个统一模板，例如：

```text
doc_id: JS002
title: 市政府办公室关于印发苏州市加快建设“人工智能+”城市行动方案（2025～2026年）的通知
region: 苏州
level: 市级
issuer: 苏州市人民政府
publish_date: 2025-09-22
doc_no: 苏府办〔2025〕102号
source_format: html
source_url: https://...

正文:
一、总体目标
……
二、重点任务
（一）……
```

### 10.4 标题处理原则

- 正文中的结构标题必须尽量保留
- 标题是后续 chunk 的天然边界信号
- 不能在清洗中把短标题行误删

---

## 11. Chunk 设计原则

### 11.1 为什么不能只按长度切

政策文档具有明显层级结构：

- 一、
- （一）
- 1.
- （1）

如果只按固定字数切，会破坏语义边界。

### 11.2 推荐方案

**先按标题分段，再按长度切分**

步骤：

1. 识别标题层级
2. 按标题分成逻辑段
3. 段过长时再做二次切分
4. 每个 chunk 保留 `title_path`

例如：

```json
{
  "chunk_id": "JS002_005",
  "doc_id": "JS002",
  "title_path": ["二、重点任务", "（一）夯实算力基础"],
  "text": "支持建设高质量算力基础设施……"
}
```

### 11.3 Chunk 应具备的信息

- chunk_id
- doc_id
- chunk_index
- text
- title_path
- metadata（地区、日期、主题等）

---

## 12. Agent 设计原则

### 12.1 V1 不求复杂多 Agent

第一版应聚焦“**单 Agent + 多工具**”。

### 12.2 推荐流程

1. 识别用户意图
   - 问答
   - 摘要
   - 比较
   - 匹配

2. 路由到相应工具或流程

3. 检索或读取相关政策 chunk

4. 生成结构化结果

### 12.3 推荐工具

- `RetrievePolicyTool`
- `SummarizePolicyTool`
- `ComparePoliciesTool`
- `MatchPolicyTool`

### 12.4 为什么这样设计

因为这能体现：

- tool calling
- route
- workflow
- state
- 结构化输出

又不会一上来复杂到不可控。

---

## 13. 第一阶段要完成的模块

## 第一阶段目标

完成 **文档处理底座 + 基础检索底座**，先不急着做花哨的 Agent UI。

### 第一阶段必须完成的模块

#### 1. metadata 读取模块
- 读取 CSV
- 建立 doc_id -> metadata 映射

#### 2. 原始文档加载模块
- PDF loader
- HTML loader
- TXT loader

#### 3. 文档清洗模块
- normalize
- 去噪
- 保留标题
- clean txt 输出

#### 4. 标题识别模块
- 识别一级、二级、三级标题

#### 5. chunk 模块
- 标题优先切分
- 长度兜底切分
- 输出 jsonl

#### 6. 向量索引构建模块
- embedding
- 入向量库
- 保存索引

#### 7. 基础检索模块
- 输入 query
- 返回 top-k chunks

#### 8. 简单 demo
- 命令行问答
- 返回答案 + 引用 chunk

### 第一阶段暂不要求

- 多轮复杂 memory
- 完整 web 前端
- 自动化全量评测平台
- 多 Agent 协作
- 复杂 rerank

---

## 14. 第一阶段建议的文件清单

### app/models
- document.py
- metadata.py
- chunk.py

### app/ingest
- metadata_loader.py
- pdf_loader.py
- html_loader.py
- txt_loader.py
- loader_factory.py

### app/clean
- cleaner.py
- normalizer.py
- rules.py
- title_detector.py

### app/chunk
- chunker.py
- title_parser.py
- splitters.py
- chunk_builder.py

### app/retrieval
- embedder.py
- vector_store.py
- retriever.py

### scripts
- clean_documents.py
- chunk_documents.py
- build_index.py
- run_demo.py

---

## 15. 第一阶段验收标准

第一阶段完成后，应至少满足以下结果：

1. 可以根据 metadata 找到原始文件
2. 可以把 PDF / HTML / TXT 统一清洗成 clean txt
3. clean txt 中保留正文结构标题
4. 可以生成结构化 chunks
5. 可以将 chunks 建索引
6. 给定 query，可以返回相关政策片段
7. 有一个简单 demo 可以展示“问答 + 引用”

---

## 16. 第二阶段方向（给 Codex 预留）

第一阶段完成后，可以继续扩展：

1. 政策比较工具
2. 企业画像匹配工具
3. LangGraph 工作流
4. session memory / summary memory
5. retrieval / answer 评测
6. FastAPI 服务层
7. 更完整的引用链路

---

## 17. 工程实施原则

1. **先做通，再做优**
2. **先做结构，再补细节**
3. **先规则清洗，再考虑 LLM 精修**
4. **先单 Agent + 多工具，再考虑多 Agent**
5. **先做离线 pipeline，再做在线服务**
6. **每层只做一件事，避免一个文件里混所有逻辑**

---

## 18. 交付物要求

交付给 Codex 的第一阶段目标应包括：

1. 项目目录结构初始化
2. 数据模型定义
3. metadata 读取
4. 文档加载
5. 文档清洗
6. chunk 切分
7. 索引构建
8. 简单检索 demo

---

## 19. 给 Codex 的开发提示

开发时请注意：

- 优先保证代码可运行、结构清晰、模块边界明确
- 不要过早引入复杂框架
- 所有路径拼接尽量基于 metadata 中的 `region`、`doc_id`、`source_format`
- 清洗脚本要可重复运行，原始文件不应被覆盖
- 标题识别必须作为 chunk 的前置步骤
- 输出要保留来源信息，便于后续引用
- 代码风格尽量偏工程项目，不要像一次性 notebook demo

---

## 20. 总结

这是一个**求职导向的 Agent 工程项目**，目标不是做最炫的技术展示，而是做一个：

- 能跑通
- 有完整链路
- 有工程结构
- 能体现 RAG + Agent + Tool + Chunk + Metadata + Eval 思维
- 能在面试中清楚讲述

的项目。

第一阶段最关键的是把 **文档数据链路和检索底座** 做扎实；  
后续再逐步叠加比较、匹配、Memory、评测和服务化能力。
