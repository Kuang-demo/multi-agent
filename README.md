# Multi-Agent Research Report System

基于 LangGraph 的多智能体研报生成系统。用户输入一个研究主题，系统自动完成规划→检索→分析→写作→审查的全链路闭环，输出带证据链和引用标记的 Markdown 研究报告。

## 架构

```
用户输入
  ↓
Planner ─→ Searcher ─→ Analyzer ─→ Writer ─→ Reviewer
             ↑                                   │
             └─── 定向补充检索（decision=research）←┘
```

| 节点 | 职责 |
|------|------|
| Planner | LLM 将主题拆成 5-6 个章节，每章 3-4 个搜索查询词 |
| Searcher | 混合检索：本地关键词 + 向量 + Tavily 网页，本地/网络分别取 top-k |
| Analyzer | LLM 基于证据逐章分析，生成摘要、要点、缺口标注 |
| Writer | LLM 生成带 `[C1][C2]` 引用标记的报告正文 |
| Reviewer | LLM-as-a-Judge 读草稿判质量，不通过则输出定向补充检索词 |

## 技术栈

- **编排**：LangGraph（StateGraph + 条件路由 + MemorySaver）
- **LLM**：DeepSeek V3（兼容 OpenAI 格式）
- **Embedding**：DashScope text-embedding-v4
- **向量库**：Chroma / **元数据**：SQLite / **网络搜索**：Tavily
- **分块策略**：结构感知三层降级（标题→段落→句子）+ 文档清洗 + 字符级重叠
- **评估**：检索评估（Hit Rate / MRR）+ 报告结构评估

## 快速开始

**1. 配置 .env**

```
DEEPSEEK_API_KEY=你的Key
DASHSCOPE_API_KEY=你的Key
TAVILY_API_KEY=你的Key    # 可选
```

**2. 安装**

```bash
pip install -r requirements.txt
```

**3. 放入知识库文档**（`.md` `.txt` `.pdf` `.docx` → `data/raw/`）

**4. 运行**

```bash
# CLI
python main.py -q "你的研究主题"

# API
uvicorn src.api:app --reload
```

## 项目结构

```
src/
├── nodes/          ← 5 个流水线节点
├── services/       ← 知识库、embedding、LLM、存储、导出
├── state.py        ← AgentState + 数据模型
├── graph.py        ← 图编排
├── config.py       ← 配置中心
└── api.py          ← FastAPI 服务
data/
├── raw/            ← 知识库源文件
├── eval/           ← 检索评估用例
├── chroma/         ← 向量持久化
└── app.db          ← SQLite
scripts/            ← 检索评估 & 生成质量评估
outputs/            ← 生成的研报
```

## API

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/health` | 健康检查 |
| POST | `/research` | 提交研究任务 |
| GET | `/knowledge-base/stats` | 知识库统计 |
| POST | `/knowledge-base/rebuild` | 重建知识库 |
| POST | `/knowledge-base/upload` | 上传文档 |

## 评估

```bash
python scripts/evaluate_retrieval.py                          # 检索质量
python scripts/evaluate_generation.py --report outputs/xxx.md # 报告结构
```
