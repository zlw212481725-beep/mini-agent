# mini-agent —— 我的第一 Agent（150 行）

> 阶段 1 产出物：一个最小的 Agent Loop，能搜我的 Obsidian 知识库并写摘要。
> 学完 AgentGuide Day 1-2 后手写，不用任何 Agent 框架——先看清 `观察 → 思考 → 动手 → 观察` 的本质。

## 架构

```
Goal（目标）
  └─> 循环（最多 5 步）:
        Think   模型看对话历史，决定下一步
        Act     调用工具: search_notes / write_summary
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

- 检索是"关键词包含匹配"，不是语义检索——这正是项目 1（RAG）要升级的点
- trace 的 cost 只记了 token 数，没乘单价
- 没有对话记忆（跑完即止）、没有人工确认环节

## 下一步

Day 3 工具设计打磨 → 阶段 2 用 RAG（向量检索 + 引用）替换 search_notes 的关键词匹配。
