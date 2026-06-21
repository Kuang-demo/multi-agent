# Multi-Agent Research Report System

基于 LangGraph 的多智能体研究报告生成系统。用户输入一个研究主题后，系统会自动完成规划、检索、分析、写作、审查和定向补充检索，最终输出带证据链和引用标记的 Markdown 报告。

这个项目的重点不是做企业级平台，而是展示一个完整的 Agent + RAG 工作流：如何拆解任务、召回证据、约束生成、审查质量，并在证据不足时回流补充。

## 工作流

```text
用户输入
  ↓
Planner → Searcher → Analyzer → Writer → Reviewer
             ↑                         │
             └── 定向补充检索 ← decision=research
```

| 节点 | 职责 |
| --- | --- |
| Planner | 将研究主题拆成可执行章节，并为每章生成 3-4 个检索查询 |
| Searcher | 构建本地知识库，执行 keyword/vector hybrid retrieval，并可接入 Tavily 网页检索 |
| Analyzer | 按章节选择高相关证据，生成摘要、要点、引用索引、证据摘录和缺口 |
| Writer | 基于章节分析生成报告正文，无证据章节不会强行调用 LLM 生成结论 |
| Reviewer | 先做规则审查，再做 LLM-as-a-Judge；不通过时输出目标章节和补充检索词 |

## 当前特性

- LangGraph `StateGraph` 编排，支持 Reviewer 条件回流。
- 统一 `AgentState`，对证据、章节分析、洞察和迭代历史做自定义合并。
- 本地知识库支持 `.md`、`.txt`、`.pdf`、`.docx`。
- 结构感知分块：Markdown 标题拆节，段落累积，超长段落退到句子级。
- 本地检索支持 keyword、vector、hybrid rerank。
- Chroma 向量库持久化，SQLite 保存 chunk 元数据和检索日志。
- Reviewer 支持 `target_section_ids`，补充检索时只搜索目标章节，减少噪声。
- LLM JSON 输出带一次轻量修复重试。
- 报告输出包含章节正文、引用索引、证据 ID、精简证据摘录和缺口说明。

## 技术栈

- **Agent 编排**：LangGraph
- **LLM**：DeepSeek OpenAI-compatible API
- **Embedding**：DashScope `text-embedding-v4`
- **Vector Store**：Chroma
- **Metadata Store**：SQLite
- **Web Search**：Tavily，可选
- **API**：FastAPI

## 快速开始

### 1. 安装依赖

建议在项目虚拟环境中安装：

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

复制 `.env.example` 为 `.env`，至少配置：

```env
DEEPSEEK_API_KEY=你的Key
DASHSCOPE_API_KEY=你的Key
TAVILY_API_KEY=你的Key    # 可选
```

常用配置：

```env
MAX_ITERATIONS=2
MIN_SECTIONS=5
MAX_SECTIONS=6
LOCAL_CHUNK_SIZE=700
LOCAL_CHUNK_OVERLAP=120
LOCAL_RETRIEVAL_TOP_K=3
VECTOR_RETRIEVAL_TOP_K=4
HYBRID_RETRIEVAL_TOP_K=4
```

### 3. 放入知识库文档

把文档放到：

```text
data/raw/
```

支持格式：

```text
.md .txt .pdf .docx
```

### 4. 运行 CLI

```bash
python main.py -q "AI Agent 在企业知识管理中的应用"
```

运行完成后，报告会写入：

```text
outputs/
```

CLI 会输出主题、最终决策、迭代次数、目标补充章节、报告路径和迭代历史。

## 知识库重建

如果修改了文档、切分策略、chunk ID 规则，建议重建知识库。

最简单方式是删除旧数据后重新运行：

```text
data/chroma/
data/app.db
```

不要删除：

```text
data/raw/
outputs/
```

也可以启动 API 后调用：

```http
POST /knowledge-base/rebuild
```

## API

启动：

```bash
uvicorn src.api:app --reload
```

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/health` | 健康检查 |
| POST | `/research` | 提交研究任务 |
| GET | `/knowledge-base/stats` | 查看知识库统计 |
| POST | `/knowledge-base/rebuild` | 重建知识库 |
| POST | `/knowledge-base/upload` | 上传知识库文档 |

`POST /research` 返回内容包括：

- `topic`
- `decision`
- `target_section_ids`
- `outline`
- `documents`
- `analyses`
- `insights`
- `review_feedback`
- `iteration_history`
- `draft`

## 项目结构

```text
src/
├── nodes/
│   ├── planner.py      # 主题规划
│   ├── searcher.py     # 本地/网页检索
│   ├── analyzer.py     # 章节分析
│   ├── writer.py       # 报告写作
│   └── reviewer.py     # 审查和回流
├── services/
│   ├── knowledge_base.py
│   ├── embeddings.py
│   ├── llm_json.py
│   ├── storage.py
│   └── report_exporter.py
├── state.py
├── graph.py
├── config.py
└── api.py

data/
├── raw/        # 原始知识库文档
├── chroma/     # Chroma 持久化数据
├── app.db      # SQLite 数据库
└── eval/       # 简单评估用例

outputs/        # 生成报告
scripts/        # 检索和报告结构评估脚本
```

## 评估脚本

项目保留了两个轻量脚本：

```bash
python scripts/evaluate_retrieval.py
python scripts/evaluate_generation.py --report outputs/xxx.md
```

它们用于基础检查，不是完整评测平台。

## 当前边界

这个项目面向学习和实习项目展示，当前没有实现：

- 多用户任务队列
- 企业级权限系统
- 复杂可观测性平台
- 大规模检索评测平台
- PostgreSQL/Redis/Kafka 等生产级基础设施

如果要扩展为企业级 Agent 平台，可以继续补异步任务、任务状态管理、工具协议化、权限隔离和更系统的评测。
