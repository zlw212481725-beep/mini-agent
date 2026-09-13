"""
mini-agent.py —— 我的第一 Agent：最小 Agent Loop
规格（对齐 AgentGuide Day 2）：
  - 2 个工具：search_notes（搜笔记）/ write_summary（写摘要）
  - 最多 8 步（Loop Controller，防无限循环烧钱；步数预算按工具数量留余量）
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

from rag_lib import semantic_search as rag_semantic_search  # 阶段2：RAG 语义检索

# ---------- 1. 配置 ----------
load_dotenv()  # 读取同目录 .env：key、接口地址、模型名、笔记目录
client = OpenAI(
    api_key=os.environ["LLM_API_KEY"],
    base_url=os.environ.get("LLM_BASE_URL"),  # 换厂商 = 改这一行 + MODEL
    timeout=30.0,   # Day3: 30 秒没响应就放弃，别干等
    max_retries=2,  # Day3: 网络抖动/限流时 SDK 自动重试 2 次
)
MODEL = os.environ["LLM_MODEL"]
NOTES_DIR = Path(os.environ.get("NOTES_DIR", "./notes"))
OUTPUT_DIR = Path("output")
TRACE_FILE = OUTPUT_DIR / "trace.jsonl"
MAX_STEPS = 8  # Loop Controller 的"下班闹钟"；步数预算 ≈ 工具调用次数 + 最终回答，留余量


# ---------- 2. 工具层（Tool Registry：真正干活的函数） ----------
def clip(text: str, limit: int = 800) -> str:
    """Day3: 工具结果统一限长——超长截断并告诉模型下一步怎么办。
    原因：模型的上下文又贵又有限，塞一大堆内容会烧钱还干扰判断。"""
    text = str(text)
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n...(结果太长已截断，原文共 {len(text)} 字；建议缩小搜索范围或分次读取)"


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


def read_note(filename: str) -> str:
    """Day3 新工具：按文件名读笔记全文（和 search_notes 配套：先搜后读）"""
    matches = [md for md in NOTES_DIR.rglob("*.md") if md.name == filename]
    if not matches:
        # Day3 原则：错误信息要能指导模型的下一步动作，而不是一句"出错了"
        return f"没有找到名为「{filename}」的笔记。可以先用 search_notes 搜关键词，再从结果里复制文件名"
    if len(matches) > 1:
        paths = "\n".join(str(m.relative_to(NOTES_DIR)) for m in matches)
        return f"有 {len(matches)} 个同名文件，请用相对路径精确指定：\n{paths}"
    return matches[0].read_text(encoding="utf-8")


def semantic_search_tool(query: str, top_k: int = 6) -> str:
    """阶段2 新工具：RAG 语义检索——按"意思"找笔记，不再死磕字面关键词"""
    try:
        return rag_semantic_search(query, top_k)
    except Exception as e:
        return f"语义检索暂不可用：{e}"   # 没建索引/没配 key 时，给模型一句人话


# 给模型看的"工具说明书"（Function Calling 标准 JSON Schema）
TOOL_SPECS = [
    {
        "type": "function",
        "function": {
            "name": "semantic_search",
            "description": "在用户的知识库里做语义搜索：按意思匹配，字面不同也能找到。调用时 query 必须用一句完整的自然语言描述要找的信息（例如：英语六级什么时候补考），不要堆砌关键词，堆关键词会降低检索质量。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "想找的内容，用自然语言描述"},
                },
                "required": ["query"],
            },
        },
    },
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
            "name": "read_note",
            "description": "读取指定笔记文件的完整内容。先用 search_notes 搜索，从结果里拿到文件名，再用本工具阅读全文。",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {"type": "string", "description": "笔记文件名，例如 用户画像.md"},
                },
                "required": ["filename"],
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
TOOL_FUNCS = {
    "semantic_search": semantic_search_tool,
    "search_notes": search_notes,
    "read_note": read_note,
    "write_summary": write_summary,
}


# ---------- 3. Context Builder：每次请求，模型能看到什么 ----------
SYSTEM_PROMPT = f"""你是一个运行在用户电脑上的笔记助手 Agent。
规则（Policy）：
1. 回答问题前，优先用 semantic_search 语义搜索用户的真实笔记（按意思匹配）；它没结果时再用 search_notes 关键词兜底；命中后想看细节，用 read_note 读全文。不许凭记忆编造笔记内容。
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
                    observation = clip(str(func(**args)))  # Day3: 统一限长再喂给模型
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
