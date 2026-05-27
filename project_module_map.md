# Policy Agent 项目模块说明

这份文档用于梳理当前项目的模块职责、数据流和系统分层，方便后续：

- 写简历项目描述
- 准备面试项目讲解
- 继续推进架构优化
- 快速定位代码职责

---

## 1. 项目整体定位

这是一个**面向政策分析场景的 RAG Agent 系统**，核心能力包括：

- 政策检索
- 单篇政策摘要
- 多文档政策汇总
- 政策对比
- 多轮上下文承接
- 回答质量评估与后续动作决策

当前项目已经从“单轮规则链”演进到：

- **LLM 驱动的 query/context resolver**
- **RAG 检索与结构化工具层**
- **多轮 memory**
- **compare / scenario advice 回答控制**

---

## 2. 系统主链

当前主链已经逐步从“关键词驱动”迁移成“结构化 resolver 驱动”。

```text
用户问题
-> context resolver
   -> contextualized_query
   -> resolved_action
   -> response_mode
   -> retrieval_goal
   -> focus
   -> answer_plan
   -> resolved_entities
-> planner（补执行默认值 / fallback）
-> rewrite
-> retrieve
-> strategy
-> summarize / multi_summary / compare / direct_answer
-> answer
-> judge
-> repair
-> next_step
```

当前这条链的核心设计思想是：

- **resolver 负责理解**
- **strategy 负责检索后分流**
- **answer 负责按模式生成回答**

---

## 3. 模块分层

### 3.1 `app/agent`

这是项目的**主编排层 / Agent 大脑**。

#### [graph.py](C:\D\Agent-learn\MyProject\app\agent\graph.py)

作用：

- 串起整条工作流
- 组织各个节点的执行顺序
- 控制第一轮主执行、repair 重试、next_step 后续动作

可以理解成：

**workflow orchestrator**

---

#### [state.py](C:\D\Agent-learn\MyProject\app\agent\state.py)

作用：

- 定义整条链路共享状态 `AgentState`
- 保存 query、retrieval、tool_output、judge、next_step 等中间结果

当前关键字段包括：

- `resolved_action`
- `intent`
- `route`
- `strategy`
- `response_mode`
- `retrieval_goal`
- `focus`
- `answer_plan`
- `rewritten_query`
- `retrieval_output`
- `tool_output`
- `final_response`

可以理解成：

**Agent 的状态总线**

---

#### [nodes.py](C:\D\Agent-learn\MyProject\app\agent\nodes.py)

作用：

- 实现每个工作流节点

主要节点包括：

- `planner_node`
- `rewrite_node`
- `retrieve_node`
- `strategy_node`
- `summarize_node`
- `summarize_policies_node`
- `compare_node`
- `answer_node`
- `judge_node`
- `repair_node`
- `next_step_node`

可以理解成：

**workflow nodes**

---

#### [planner.py](C:\D\Agent-learn\MyProject\app\agent\planner.py)

作用：

- 原本是主理解节点
- 现在逐步退化成：
  - 执行默认值补全器
  - fallback 规划器

当前主要负责补：

- `needs_rag`
- `needs_rewrite`
- `answer_style`
- 初始 `route`

---

#### [rewrite.py](C:\D\Agent-learn\MyProject\app\agent\rewrite.py)

作用：

- 把用户问题改写成更适合检索的 query

输出：

- `primary_query`
- `alternative_queries`
- `keywords`
- `rewrite_reason`

---

#### [strategy.py](C:\D\Agent-learn\MyProject\app\agent\strategy.py)

作用：

- retrieval 后决定下游该走哪条处理路径

当前主要分成：

- `direct_answer`
- `single_doc_summary`
- `multi_doc_summary`
- `compare`

新的方向是：

- 优先根据 `retrieval_goal`
- 再结合 retrieval 结果校正

而不是只靠 query 关键词。

---

#### [answer.py](C:\D\Agent-learn\MyProject\app\agent\answer.py)

作用：

- 基于工具证据生成最终回答

当前已经支持：

- 普通检索回答
- 结构化摘要
- 普通 compare
- 场景化 compare 建议

并开始依赖：

- `response_mode`
- `focus`
- `answer_plan`

而不只是 query 本身。

---

#### [judge.py](C:\D\Agent-learn\MyProject\app\agent\judge.py)

作用：

- 对当前回答做质量评估

输出：

- `verdict`
- `score`
- `reason`
- `followup`

---

#### [repair.py](C:\D\Agent-learn\MyProject\app\agent\repair.py)

作用：

- 回答不够好时生成一次修复性 retry 决策

包括：

- 要不要 retry
- 新 query 是什么
- repair strategy 是什么

---

#### [next_step.py](C:\D\Agent-learn\MyProject\app\agent\next_step.py)

作用：

- 当前轮回答结束后，决定下一步：
  - 结束
  - 追问
  - 建议 compare
  - 自动 route switch

---

#### [router.py](C:\D\Agent-learn\MyProject\app\agent\router.py)

作用：

- 旧的规则路由器
- 当前主要是 fallback / 兼容层

长期目标是弱化它在主链中的地位。

---

### 3.2 `app/memory`

这是项目的**多轮上下文与会话记忆层**。

#### [session.py](C:\D\Agent-learn\MyProject\app\memory\session.py)

定义：

- `SessionTurn`
- `WorkingMemory`
- `MemoryEntity`
- `ComparisonMemory`
- `SessionMemory`

作用：

- 保存原始对话历史
- 保存当前任务状态
- 保存最近对象
- 保存当前比较对象组

---

#### [store.py](C:\D\Agent-learn\MyProject\app\memory\store.py)

作用：

- 进程内 session store
- 根据 `session_id` 复用同一段会话 memory

---

#### [updater.py](C:\D\Agent-learn\MyProject\app\memory\updater.py)

作用：

- 每轮执行结束后刷新 memory

负责：

- 追加 turns
- 更新 working_memory
- 构建 recent_entities
- 维护 active_comparison
- 累积对象记忆

---

#### [completion.py](C:\D\Agent-learn\MyProject\app\memory\completion.py)

作用：

- 当前最关键的 memory 节点
- 对用户问题做：
  - 上下文补全
  - 指代消解
  - 比较对象扩展
  - 回答模式判断
  - 回答计划生成

当前结构化输出包括：

- `contextualized_query`
- `resolved_action`
- `response_mode`
- `retrieval_goal`
- `focus`
- `answer_plan`
- `resolved_entities`

这是现在的：

**context resolver**

---

### 3.3 `app/tools`

这是项目的**证据处理工具层**。

#### [retrieve_policy.py](C:\D\Agent-learn\MyProject\app\tools\retrieve_policy.py)

作用：

- 统一检索入口
- 返回 top-k 政策证据

---

#### [summarize_policy.py](C:\D\Agent-learn\MyProject\app\tools\summarize_policy.py)

作用：

- 单篇政策摘要
- 围绕单个 doc 读取全部 chunk
- 提取：
  - 政策概览
  - 支持重点
  - 适用对象
  - 申报条件

---

#### [summarize_policies.py](C:\D\Agent-learn\MyProject\app\tools\summarize_policies.py)

作用：

- 多政策汇总
- 按 doc 聚合检索结果
- 逐篇摘要
- 再按 section 汇总

---

#### [compare_policy.py](C:\D\Agent-learn\MyProject\app\tools\compare_policy.py)

作用：

- 双对象政策对比
- 先各自摘要，再按 section 并排对比

当前局限：

- 仍然偏双对象 compare
- 后续如果要更强，需要升级到对象组 compare

---

#### [compare_policies.py](C:\D\Agent-learn\MyProject\app\tools\compare_policies.py)

作用：

- 更偏多政策 compare 的辅助层 / 扩展入口

---

#### [match_policy.py](C:\D\Agent-learn\MyProject\app\tools\match_policy.py)

作用：

- 匹配 / 推荐类任务的预留入口

---

### 3.4 `app/retrieval`

这是项目的**RAG 检索层**。

#### [retriever.py](C:\D\Agent-learn\MyProject\app\retrieval\retriever.py)

作用：

- 构建和加载检索器
- 恢复 FAISS / payload / manifest
- 执行向量检索

---

#### [embedder.py](C:\D\Agent-learn\MyProject\app\retrieval\embedder.py)

作用：

- embedding 模型封装

---

#### [vector_store.py](C:\D\Agent-learn\MyProject\app\retrieval\vector_store.py)

作用：

- 向量库封装
- 当前主要是 FAISS

---

#### [reranker.py](C:\D\Agent-learn\MyProject\app\retrieval\reranker.py)

作用：

- rerank 扩展入口

---

### 3.5 `app/ingest`

这是项目的**数据接入层**。

#### [metadata_loader.py](C:\D\Agent-learn\MyProject\app\ingest\metadata_loader.py)

作用：

- 读取 [policies.csv](C:\D\Agent-learn\MyProject\data\metadata\policies.csv)

---

#### [loader_factory.py](C:\D\Agent-learn\MyProject\app\ingest\loader_factory.py)

作用：

- 根据 metadata 的 `source_format` 选择：
  - txt loader
  - pdf loader

并支持：

- 同一 doc_id 多源并存时按 metadata 指定主源

---

#### [pdf_loader.py](C:\D\Agent-learn\MyProject\app\ingest\pdf_loader.py)

作用：

- PDF 正文抽取
- 支持 `pypdf`
- 支持 `PyMuPDF` fallback

---

#### [txt_loader.py](C:\D\Agent-learn\MyProject\app\ingest\txt_loader.py)

作用：

- 读取官方正文 txt

---

#### [html_loader.py](C:\D\Agent-learn\MyProject\app\ingest\html_loader.py)

作用：

- HTML 数据源的扩展入口

---

#### [source_audit.py](C:\D\Agent-learn\MyProject\app\ingest\source_audit.py)

作用：

- 核对 metadata 和 raw 文件的一致性
- 判断哪些文档已具备迁移到 txt 的条件

---

### 3.6 `app/clean`

这是项目的**文本清洗层**。

#### [cleaner.py](C:\D\Agent-learn\MyProject\app\clean\cleaner.py)

- 清洗入口

#### [normalizer.py](C:\D\Agent-learn\MyProject\app\clean\normalizer.py)

- 文本标准化

#### [rules.py](C:\D\Agent-learn\MyProject\app\clean\rules.py)

- 清洗规则集合

#### [title_detector.py](C:\D\Agent-learn\MyProject\app\clean\title_detector.py)

- 标题识别

---

### 3.7 `app/chunk`

这是项目的**切片层**。

#### [chunk_builder.py](C:\D\Agent-learn\MyProject\app\chunk\chunk_builder.py)

作用：

- 从 raw 数据 + metadata 构建 chunk
- 导出 `policy_chunks.jsonl`

---

#### [chunker.py](C:\D\Agent-learn\MyProject\app\chunk\chunker.py)

- 实际切片逻辑

#### [title_parser.py](C:\D\Agent-learn\MyProject\app\chunk\title_parser.py)

- 层级标题解析

#### [splitters.py](C:\D\Agent-learn\MyProject\app\chunk\splitters.py)

- 句段切分规则

---

### 3.8 `app/api`

这是项目的**服务层 / 验证入口**。

#### [server.py](C:\D\Agent-learn\MyProject\app\api\server.py)

作用：

- FastAPI 服务入口
- `/ask` 接口
- 接入：
  - session memory
  - context resolver
  - run_agent_workflow

---

#### [index.html](C:\D\Agent-learn\MyProject\app\api\static\index.html)

作用：

- 前端验证页
- 支持连续对话
- 支持调试展示：
  - `resolved_action`
  - `response_mode`
  - `retrieval_goal`
  - `focus`
  - `context_resolution`
  - `session_memory`

---

### 3.9 `app/llm`

这是项目的**模型调用层**。

#### [client.py](C:\D\Agent-learn\MyProject\app\llm\client.py)

作用：

- OpenAI / 云雾兼容客户端
- 提供：
  - `generate_text`
  - `parse_structured_response`

这是所有 LLM 节点的调用底座。

---

### 3.10 `scripts`

这是项目的**工程辅助层**。

#### [rebuild_data.py](C:\D\Agent-learn\MyProject\scripts\rebuild_data.py)

作用：

- 正式重建：
  - chunk
  - retrieval payload
  - FAISS index
  - manifest

这是当前项目很关键的工程入口。

---

### 3.11 `tests`

这是项目的**回归验证层**。

当前已经覆盖：

- graph 主链流转
- llm client
- llm nodes
- source audit / loader
- memory / memory completion
- retrieve
- strategy
- summarize rules
- answer prompt

---

## 4. 当前架构重点

如果只挑最值得你在面试里讲的 5 个重点，可以压成：

1. **LLM resolver 驱动的主理解层**
   - 不再主要依赖 query 关键词
   - 支持多轮上下文补全、指代消解、回答模式判断

2. **结构化 AgentState**
   - `resolved_action / retrieval_goal / response_mode / focus / answer_plan`
   - 让理解、检索目标、回答控制分层

3. **RAG + 工具层**
   - retrieve / summarize / multi_summary / compare
   - 不是单纯聊天，而是任务型政策分析

4. **多轮 memory**
   - session history
   - working memory
   - entity memory
   - comparison memory

5. **数据与工程化**
   - txt/pdf 多源支持
   - 数据审计
   - 一键重建 chunk / 索引
   - 测试覆盖

---

## 5. 当前仍然存在的限制

1. compare 工具本身还偏双对象
2. `router.py` / `strategy.py` 仍保留部分 fallback 规则
3. `memory/updater.py` 里还有少量轻规则推断
4. 多对象 compare 已开始支持，但还不是完整的对象组 compare 系统

---

## 6. 一句话总结

这个项目已经不是“调 API 做聊天”，而是一个：

**包含数据接入、文本清洗、切片、向量检索、结构化工具、多轮 memory、LLM resolver、回答控制、测试与工程辅助脚本的政策分析型 RAG Agent 系统。**

