# Policy Agent

面向公开政策文档的 `RAG + Agent` 项目，当前聚焦人工智能政策的：

- 政策检索
- 单篇政策摘要
- 多文档政策汇总
- 政策对比

项目同时支持：

- 纯规则链路运行
- 接入 OpenAI 兼容 LLM 进行增强

## 当前架构

当前主链路已经从“前置硬路由”重构成“resolver 主理解 + 统一先检索，再按证据分流”。

```text
用户问题
-> context resolver
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

几个关键概念：

- `resolved_action`
  代表 resolver 对当前轮任务的主理解，例如 retrieve / summarize / compare

- `route`
  代表当前这一跳先执行什么，当前首跳统一为 `retrieve`

- `strategy`
  代表检索完成后，系统最终选择哪种处理方案

- `response_mode`
  代表最终回答组织方式，例如普通对比或场景化建议

- `retrieval_goal`
  代表这一轮检索真正想召回什么对象，例如单篇政策、地区多政策、地区比较等

## 四种策略

### 1. `direct_answer`

适合普通问答，例如：

- `上海有哪些 AI 政策？`
- `哪些政策提到了算力券？`

### 2. `single_doc_summary`

适合单篇政策摘要，例如：

- `请总结《上海市进一步扩大人工智能应用的若干措施》`

### 3. `multi_doc_summary`

适合地区 / 主题 / 时间范围汇总，例如：

- `总结一下上海的 AI 政策`
- `梳理近两年的人工智能支持政策`

### 4. `compare`

适合对比型请求，例如：

- `比较北京和上海的大模型政策`

## 目录说明

```text
app/
  agent/        Agent 编排、状态流、策略选择
  chunk/        切片逻辑
  clean/        清洗逻辑
  ingest/       文档与元数据加载
  llm/          OpenAI 兼容 LLM 客户端
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
- [nodes.py](c:/D/Agent-learn/MyProject/app/agent/nodes.py)：节点实现
- [state.py](c:/D/Agent-learn/MyProject/app/agent/state.py)：共享状态
- [strategy.py](c:/D/Agent-learn/MyProject/app/agent/strategy.py)：检索后策略分流
- [retrieve_policy.py](c:/D/Agent-learn/MyProject/app/tools/retrieve_policy.py)：统一检索
- [summarize_policy.py](c:/D/Agent-learn/MyProject/app/tools/summarize_policy.py)：单篇摘要
- [summarize_policies.py](c:/D/Agent-learn/MyProject/app/tools/summarize_policies.py)：多文档汇总
- [compare_policy.py](c:/D/Agent-learn/MyProject/app/tools/compare_policy.py)：对比工具
- [client.py](c:/D/Agent-learn/MyProject/app/llm/client.py)：OpenAI / 云雾兼容客户端

## 当前状态流

`AgentState` 是整条链路的共享容器，重点字段包括：

- `query`
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
- `judge_verdict`
- `next_step_action`

其中：

- `retrieval_output` 保存统一检索得到的原始结果
- `tool_output` 保存当前真正交给 `answer / judge / repair / next_step` 消费的工具输出

当前 `tool_output` 主要有四种类型：

- `RetrievePolicyOutput`
- `PolicySummaryOutput`
- `MultiPolicySummaryOutput`
- `PolicyCompareOutput`

## 无 LLM 运行

如果未配置可用 LLM，系统会自动退回规则链路：

- 规则 planner
- 规则 rewrite
- 规则 answer
- 规则 judge
- 规则 repair
- 规则 next_step

这条链路当前已经可以独立跑通。

## LLM 接入

当前项目已经支持接入 OpenAI 兼容接口，例如云雾 API。

### 支持的环境变量

最低需要：

```env
YUNWU_API_KEY=your_key
```

可选：

```env
YUNWU_BASE_URL=https://yunwu.ai/v1

PLANNER_MODEL=gpt-5.4
REWRITE_MODEL=gpt-5.4-mini
ANSWER_MODEL=gpt-5.4
JUDGE_MODEL=gpt-5.4-mini
NEXT_STEP_MODEL=gpt-5.4-mini
REPAIR_MODEL=gpt-5.4-mini
```

也兼容 OpenAI 风格变量：

```env
OPENAI_API_KEY=
OPENAI_BASE_URL=
OPENAI_MODEL=
```

### 示例配置文件

仓库已提供：

- [.env.example](c:/D/Agent-learn/MyProject/.env.example)

### 当前建议的模型分配

- `planner = gpt-5.4`
- `rewrite = gpt-5.4-mini`
- `answer = gpt-5.4`
- `judge = gpt-5.4-mini`
- `next_step = gpt-5.4-mini`
- `repair = gpt-5.4-mini`

### 当前已确认接通的节点

真实链路里已经验证过：

- `planner_source = llm`
- `answer_source = llm`
- `judge_source = llm`
- `next_step_source = llm`

`rewrite` 和 `repair` 已接入模型能力；是否实际触发，取决于具体问题是否进入对应分支。

## 运行方式

CLI 入口：

- [app/main.py](c:/D/Agent-learn/MyProject/app/main.py)

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

你也可以用根目录下的：

- [main.py](c:/D/Agent-learn/MyProject/main.py)

先做单独的云雾连通性测试。

## Web 验证页

项目现在提供了一个最小可用的 Web 验证页，便于快速测试：

- 输入问题
- 查看最终回答
- 查看 `resolved_action / response_mode / retrieval_goal / focus / route / strategy / judge / next_step`
- 查看完整状态 JSON

服务端入口：

- [server.py](c:/D/Agent-learn/MyProject/app/api/server.py)

前端页面：

- [index.html](c:/D/Agent-learn/MyProject/app/api/static/index.html)

启动方式：

```powershell
uvicorn app.api.server:app --reload
```

启动后打开：

```text
http://127.0.0.1:8000
```

## 数据重建

当你修改这些内容后：

- `data/raw/...`
- `data/metadata/policies.csv`
- 清洗 / chunk / 检索相关代码

需要先重建 chunk 与检索索引，再做验证。

正式重建入口：

- [rebuild_data.py](c:/D/Agent-learn/MyProject/scripts/rebuild_data.py)

全量重建：

```powershell
python scripts/rebuild_data.py
```

只做局部 chunk 调试（不会覆盖正式索引）：

```powershell
python scripts/rebuild_data.py --chunks-only --doc-ids BJ001 BJ002
```

## 测试

当前测试基于 `unittest`，可直接执行：

```powershell
python -m unittest discover -s tests -p "test_*.py" -v
```

当前已覆盖：

- 检索工具封装
- 检索后策略选择
- graph 主链路四种主要分支
- 单篇摘要规则抽取的关键行为

## 评测

项目提供了最小可用评测集与评测脚本：

- [cases.json](c:/D/Agent-learn/MyProject/app/eval/cases.json)
- [run_eval.py](c:/D/Agent-learn/MyProject/app/eval/run_eval.py)

运行全部评测：

```powershell
python app/eval/run_eval.py
```

只抽样运行部分 case：

```powershell
python app/eval/run_eval.py --case-ids compare_bj_sh_model scenario_compare_bj_sh
```

评测报告输出到：

- [eval_report.json](c:/D/Agent-learn/MyProject/outputs/eval_report.json)

## 依赖

见：

- [requirements.txt](c:/D/Agent-learn/MyProject/requirements.txt)

安装示例：

```powershell
pip install -r requirements.txt
```

## 当前限制

- 部分 PDF 文档 OCR 质量较差，会影响规则摘要和 compare 质量
- compare 目前更偏“双文档对比”，还可以继续升级为“对象分组 compare”
- memory 还没有真正做成会话级能力

## 后续方向

- 补会话级 memory
- 继续优化 PDF / OCR 清洗质量
- 增强 compare 为分组式比较
- 视需要补 FastAPI 或 Web demo
