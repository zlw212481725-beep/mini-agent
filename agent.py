"""
mini-agent.py —— 我的第一 Agent：最小 Agent Loop
规格（对齐 AgentGuide Day 2）：
  - 2 个工具：search_notes（搜笔记）/ write_summary（写摘要）
  - 最多 5 步（Loop Controller，防无限循环烧钱）
  - 每步写 JSONL trace（可审计、可复盘）
  - 出错不崩溃，把可读的错误喂回给模型或返回给用户

跑法：python agent.py "在知识库里找六级相关的计划并写摘要"
"""
import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

# ---------- 1. 配置 ----------
load_dotenv()  # 读取同目录 .env：key、接口地址、模型名、笔记目录
client = OpenAI(
    api_key=os.environ["LLM_API_KEY"],
    base_url=os.environ.get("LLM_BASE_URL"),  # 换厂商 = 改这一行 + MODEL
)
MODEL = os.environ["LLM_MODEL"]
NOTES_DIR = Path(os.environ.get("NOTES_DIR", "./notes"))
OUTPUT_DIR = Path("output")
TRACE_FILE = OUTPUT_DIR / "trace.jsonl"
MAX_STEPS = 5  # Loop Controller 的"下班闹钟"


# ---------- 2. 工具层（Tool Registry：真正干活的函数） ----------
def search_notes(query: str, max_hits: int = 8) -> str:
    """在笔记目录所有 .md 文件里按关键词搜，返回 '文件名:行号: 内容' 列表"""
    hits = []
    for md in NOTES_DIR.rglob("*.md"):
        if len(hits) >= max_hits:
            break
        try:
            lines = md.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeDecodeError):
            continue  # 读不了的文件跳过，不崩溃
        for i, line in enumerate(lines, 1):
            if query.lower() in line.lower():
                hits.append(f"{md.name} 第{i}行: {line.strip()[:150]}")
                if len(hits) >= max_hits:
                    break
    return "\n".join(hits) if hits else f"笔记里没找到包含「{query}」的内容"


def write_summary(text: str) -> str:
    """把摘要文本写入 output/ 目录，返回保存路径"""
    OUTPUT_DIR.mkdir(exist_ok=True)
    path = OUTPUT_DIR / f"summary_{int(time.time())}.md"
    path.write_text(text, encoding="utf-8")
    return f"摘要已保存到 {path}"


# 给模型看的"工具说明书"（Function Calling 标准 JSON Schema）
TOOL_SPECS = [
    {
        "type": "function",
        "function": {
            "name": "search_notes",
            "description": "在用户的 Obsidian 知识库里搜索笔记，输入关键词，返回匹配的原文行。回答任何问题前都应该先搜一次。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "要搜索的关键词"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_summary",
            "description": "把整理好的总结文本保存成文件，方便用户以后查看。",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "要保存的总结全文"},
                },
                "required": ["text"],
            },
        },
    },
]
TOOL_FUNCS = {"search_notes": search_notes, "write_summary": write_summary}


# ---------- 3. Context Builder：每次请求，模型能看到什么 ----------
SYSTEM_PROMPT = f"""你是一个运行在用户电脑上的笔记助手 Agent。
规则（Policy）：
1. 回答问题前，先用 search_notes 查用户的真实笔记，不许凭记忆编造笔记内容。
2. 用户要总结/存档时，用 write_summary 保存。
3. 笔记里查不到就明说"笔记里没有相关内容"，不许猜。
4. 你最多执行 {MAX_STEPS} 步，动作要节约。"""


# ---------- 4. Trace：每步留痕，写进 JSONL ----------
def dump_trace(records: list):
    with TRACE_FILE.open("a", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ---------- 5. Loop Controller：最小 Agent Loop ----------
def run_agent(goal: str) -> str:
    print(f"🎯 目标: {goal}\n")
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": goal},
    ]
    trace_records = []
    OUTPUT_DIR.mkdir(exist_ok=True)

    for step in range(1, MAX_STEPS + 1):
        try:
            resp = client.chat.completions.create(
                model=MODEL, messages=messages, tools=TOOL_SPECS
            )
        except Exception as e:
            return f"❌ 第 {step} 步调用模型失败: {e}"

        msg = resp.choices[0].message
        tokens = resp.usage.total_tokens if resp.usage else 0

        if not msg.tool_calls:  # 模型不再调工具 => 它认为任务完成
            trace_records.append({
                "step": step, "thought_summary": (msg.content or "")[:200],
                "tool": "FINAL_ANSWER", "args": None, "observation": None,
                "cost_estimate_tokens": tokens,
            })
            dump_trace(trace_records)
            return msg.content or "（模型没有返回内容）"

        messages.append(msg)  # 把"我要调工具"的决定记回对话
        for call in msg.tool_calls:
            name = call.function.name
            try:
                args = json.loads(call.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            print(f"  [第{step}步] 🔧 {name}({args})")

            func = TOOL_FUNCS.get(name)
            if func is None:
                observation = f"错误：没有名为 {name} 的工具"
            else:
                try:
                    observation = str(func(**args))
                except Exception as e:
                    observation = f"工具执行出错: {e}"  # 出错不崩溃，喂回给模型自己想办法

            print(f"          -> {observation[:100]}")
            trace_records.append({
                "step": step, "thought_summary": (msg.content or "")[:200],
                "tool": name, "args": args, "observation": observation[:200],
                "cost_estimate_tokens": tokens,
            })
            messages.append({
                "role": "tool", "tool_call_id": call.id, "content": observation,
            })

    dump_trace(trace_records)
    return f"⚠️ 达到最大步数 {MAX_STEPS} 仍未完成。中间过程见 {TRACE_FILE}"


# ---------- 6. 入口 ----------
if __name__ == "__main__":
    goal = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "帮我搜搜知识库里有没有关于六级考试的计划，写个摘要保存"
    answer = run_agent(goal)
    print(f"\n💬 最终回答:\n{answer}")
