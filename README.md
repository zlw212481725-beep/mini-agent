# mini-agent —— 我的第一 Agent（150 行）

> 阶段 1 产出物：一个最小的 Agent Loop，能搜我的 Obsidian 知识库并写摘要。
> 学完 AgentGuide Day 1-2 后手写，不用任何 Agent 框架——先看清 `观察 → 思考 → 动手 → 观察` 的本质。

## 架构

```
Goal（目标）
  └─> 循环（最多 8 步，按工具数留余量）:
        Think   模型看对话历史，决定下一步
        Act     调用工具: semantic_search / search_notes / read_note / write_summary
        Observe 把工具结果记回对话
        Stop    模型不再调工具 = 任务完成；或步数用完强制停
  └─> 每步写 trace.jsonl（可审计）
```

8 个模块的落点：Goal/Policy = system prompt；State = messages 列表；
Tool Registry = TOOL_FUNCS；Context Builder = SYSTEM_PROMPT + messages 组装；
Loop Controller = MAX_STEPS + for 循环；Eval/Trace = output/trace.jsonl。

## 运行

```bash
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt   # 已装好可跳过
.venv\Scripts\python agent.py "在知识库里找六级相关的计划并写个摘要"
```

配置在 `.env`（不入库）：`LLM_API_KEY` / `LLM_BASE_URL` / `LLM_MODEL` / `NOTES_DIR`。
换厂商只需改 BASE_URL 和 MODEL（OpenAI 兼容接口通用）。

## 验收任务（5 条固定）

1. 搜"六级"并写摘要存档 → 应调用 search_notes + write_summary
2. 搜"秋招"，口述时间线 → 应引用笔记原文行
3. 搜"健身" → 返回笔记里的真实内容
4. 搜一个不存在的词 → 应回答"笔记里没有"，不许编造
5. 问一个笔记里查不到的 Agent 概念 → 应明说笔记里没有，然后可以凭常识答

## 已知限制（诚实清单）

- 语义检索对"关键词堆叠式"查询仍会被单字带偏（如"计划"），标准解法是 LLM 查询改写 + rerank——项目 1 的进阶课题
- 切块是"标题分段 + 滑窗"，按语义切分（embedding 分段）是更聪明的切法，留作进阶
- trace 的 cost 只记了 token 数，没乘单价
- 没有对话记忆（跑完即止）、没有人工确认环节（进阶课题）

## Day 3 升级记录（2026-09-13）

- `clip()`：所有工具结果统一限长 800 字，超长截断并提示模型下一步怎么办（控上下文成本、防干扰）
- `read_note(filename)`：第三个工具，补全"先搜后读"闭环；错误信息会指导模型的下一步动作
- API 客户端加 `timeout=30` + `max_retries=2`：网络抖动/限流不再让程序直接死掉
- 实战调参：任务超过 5 步被强制收工后，按"工具数 + 最终回答 + 余量"把步数预算调到 8

## 阶段 2 升级记录：RAG 语义检索上线（2026-09-13）

- `rag_lib.py`：RAG 四步链路——切块 → 向量化（BAAI/bge-m3，1024 维）→ Chroma 本地向量库（cosine）→ 语义检索
- 切块策略 v2：**按 Markdown 标题切主题块**（一块只说一件事），段内超长再滑窗（重叠 80 字防截断）
- Agent 新增第 4 个工具 `semantic_search`，军规改为"**语义优先、关键词兜底**"
- `build_index.py`：一键把知识库切块入库（upsert 幂等，笔记大改后重跑即可）
- 调试实录（面试素材）：v1 固定滑窗把目标块稀释到前三开外 → 改主题切块后进入前三；关键词堆叠式查询带偏排名 → 工具说明书要求模型**用完整自然句查询**（查询改写）后，Agent 实际调用全部使用自然语言问句

## 下一步

Day 3 工具设计打磨 → 阶段 2 用 RAG（向量检索 + 引用）替换 search_notes 的关键词匹配。
