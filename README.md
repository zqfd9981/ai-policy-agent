# Policy Agent

面向公开政策文档的 `RAG + Agent` 项目，当前聚焦人工智能政策的检索、单篇摘要、多文档汇总和政策对比。

## 项目目标

- 支持 `PDF / TXT` 政策文档接入、清洗、切片、向量检索
- 基于统一检索入口完成政策问答
- 支持单篇政策摘要
- 支持地区/主题级多文档政策汇总
- 支持双政策对比
- 保留可解释的引用与状态流，便于调试和后续扩展

## 当前主链路

当前版本已经从“前置硬路由”改成“统一先检索，再按证据分流”。

整体流程如下：

```text
用户问题
-> planner
-> rewrite
-> retrieve
-> strategy selector
   -> direct_answer
   -> single_doc_summary
   -> multi_doc_summary
   -> compare
-> answer
-> judge
-> repair
-> next_step
```

核心含义：

- `intent`：用户想做什么，例如普通问答、摘要、对比
- `route`：当前这一步先执行什么，当前首跳统一为 `retrieve`
- `strategy`：检索完成后，系统最终选择哪种处理方案

## 四种策略

### 1. `direct_answer`

适合普通政策问答，例如：

- `上海有哪些 AI 政策？`
- `哪些政策提到了算力券？`

执行方式：

- 检索 `top-k chunk`
- 直接组织回答

### 2. `single_doc_summary`

适合单篇政策摘要，例如：

- `请总结<某政策标题>`

执行方式：

- 先检索
- 检索结果明显集中到单篇文档时，切到单篇摘要
- 对目标文档的全部 chunk 做结构化摘要

### 3. `multi_doc_summary`

适合地区/主题/时间范围汇总，例如：

- `总结一下上海的 AI 政策`
- `梳理近两年的人工智能支持政策`

执行方式：

- 先检索
- 聚合多篇高相关政策
- 对每篇做单篇摘要
- 再做跨文档 section 汇总

### 4. `compare`

适合对比型请求，例如：

- `比较北京和上海的大模型政策`

执行方式：

- 先检索
- 识别两组或两篇候选政策
- 分别摘要
- 再做结构化对比

## 目录说明

```text
app/
  agent/        Agent 编排、状态流、策略选择
  chunk/        切片逻辑
  clean/        清洗逻辑
  ingest/       文档与元数据加载
  llm/          OpenAI 调用封装
  models/       数据结构
  retrieval/    embedding、FAISS、检索器
  tools/        检索、摘要、汇总、对比工具
tests/          单元测试与流程测试
configs/        配置文件
data/           原始数据与元数据
outputs/        chunk、检索索引等产物
```

## 关键模块

- [graph.py](c:/D/Agent-learn/MyProject/app/agent/graph.py)：主编排入口
- [nodes.py](c:/D/Agent-learn/MyProject/app/agent/nodes.py)：各节点实现
- [state.py](c:/D/Agent-learn/MyProject/app/agent/state.py)：共享状态
- [strategy.py](c:/D/Agent-learn/MyProject/app/agent/strategy.py)：检索后策略选择
- [retrieve_policy.py](c:/D/Agent-learn/MyProject/app/tools/retrieve_policy.py)：统一检索工具
- [summarize_policy.py](c:/D/Agent-learn/MyProject/app/tools/summarize_policy.py)：单篇摘要
- [summarize_policies.py](c:/D/Agent-learn/MyProject/app/tools/summarize_policies.py)：多文档汇总
- [compare_policy.py](c:/D/Agent-learn/MyProject/app/tools/compare_policy.py)：对比工具

## 当前数据流

`AgentState` 是整条链路的核心容器，里面最重要的字段有：

- `query`
- `intent`
- `route`
- `strategy`
- `rewritten_query`
- `retrieval_output`
- `tool_output`
- `final_response`
- `judge_verdict`
- `next_step_action`

其中：

- `retrieval_output` 保留统一检索得到的原始结果
- `tool_output` 表示当前真正交给 `answer / judge / repair / next_step` 消费的工具输出

`tool_output` 目前主要有四种类型：

- `RetrievePolicyOutput`
- `PolicySummaryOutput`
- `MultiPolicySummaryOutput`
- `PolicyCompareOutput`

## 运行方式

CLI 入口在 [main.py](c:/D/Agent-learn/MyProject/app/main.py)。

示例：

```powershell
python app\main.py 上海有哪些AI政策
python app\main.py 总结一下上海的AI政策
python app\main.py 比较北京和上海的大模型政策
```

如果想看完整状态：

```powershell
python app\main.py 总结一下上海的AI政策 --json
```

## 测试

当前测试基于 `unittest`，可直接执行：

```powershell
python -m unittest discover -s tests -p "test_*.py" -v
```

当前已覆盖：

- 检索工具封装
- 检索后策略选择
- graph 主链路的四种主要分支

## 当前已知限制

- 项目说明文档里仍存在部分编码异常文本
- 还没有正式依赖清单文件，例如 `requirements.txt` 或 `pyproject.toml`
- `compare` 目前仍偏“双文档对比”，后续可继续增强为“对象分组式对比”
- `multi_doc_summary` 的候选文档筛选还可以进一步引入更强的地区、主题、时间过滤

## 下一步建议

- 补依赖清单与环境说明
- 增加真实样例的回归测试
- 优化多文档汇总的候选文档选择逻辑
- 优化 compare 为分组比较模式
