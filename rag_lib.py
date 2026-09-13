"""
rag_lib.py —— RAG 三件套：切块 / 向量化 / 检索（阶段 2 核心库）

RAG 链路（检索增强生成）：
  建索引：md 文件 → 切成小块(chunk) → 每块变成"意思坐标"(embedding) → 存进向量库(Chroma)
  查询时：问题 → 也变成坐标 → 找坐标距离最近的几块 → 喂给模型当参考

为什么能按"意思"找？—— embedding 把每段文字变成一串数字（坐标），
意思相近的文字，坐标也相近。所以"六级考试计划"能搜到"12月可补考六级"，
关键词搜索就做不到（一个字都对不上）。
"""
import os
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

NOTES_DIR = Path(os.environ.get("NOTES_DIR", "./notes"))
CHROMA_DIR = Path("chroma_db")   # 向量库的存盘位置（会自动生成，别手改）
COLLECTION = "vault"


# ---------- 第 1 步：切块 ----------
def chunk_text(text: str, size: int = 500, overlap: int = 80) -> list:
    """把长文本切块。v2 策略：先按 Markdown 标题切（每块=一个主题），
    单块仍超长再用滑窗细分（块间重叠 overlap 字防截断）。

    为什么按标题切？—— v1 的固定滑窗会把"六级"那一行和一堆无关内容
    切进同一块，整块坐标被稀释，检索排名就掉。按主题切，一块只说一件事。
    （实战教训：这是调试"检索不准"时用真金白银换来的结论）
    """
    import re
    sections = re.split(r"\n(?=#{1,4} )", text)     # 在标题行前面断开
    chunks = []
    for sec in sections:
        sec = sec.strip()
        if len(sec) < 20:                           # 太碎的（如纯 frontmatter）跳过
            continue
        if len(sec) <= size:
            chunks.append(sec)
            continue
        start = 0
        while start < len(sec):                     # 段内超长 → 滑窗
            chunks.append(sec[start:start + size])
            start += size - overlap
    return chunks


# ---------- 第 2 步：向量化（文字 → 意思坐标） ----------
def get_embed_client() -> OpenAI:
    """拿到向量化 API 的客户端。硅基流动 / 百炼都是 OpenAI 兼容接口。"""
    key = os.environ.get("EMBED_API_KEY", "")
    if not key or "在这里填" in key:
        raise RuntimeError(
            "向量化 API 还没配置：去 cloud.siliconflow.cn 注册（免费），"
            "在 API 密钥页新建一个 key，填进 .env 的 EMBED_API_KEY"
        )
    return OpenAI(api_key=key, base_url=os.environ["EMBED_BASE_URL"])


def embed_texts(texts: list, client=None, batch: int = 32) -> list:
    """把一批文本变成坐标列表。一次请求打包 32 条，省请求次数。"""
    client = client or get_embed_client()
    model = os.environ.get("EMBED_MODEL", "BAAI/bge-m3")
    vectors = []
    for i in range(0, len(texts), batch):
        resp = client.embeddings.create(model=model, input=texts[i:i + batch])
        vectors.extend(d.embedding for d in resp.data)
        print(f"    已向量化 {min(i + batch, len(texts))}/{len(texts)} 块")
    return vectors


# ---------- 第 3 步：向量库（存坐标 + 找最近） ----------
def get_collection():
    """打开（没有就创建）向量库。cosine = 用夹角算相似，坐标方向越近意思越像。"""
    db = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return db.get_or_create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})


def build_index():
    """把整个知识库切块、向量化、入库。跑一次即可；笔记大改后重跑（upsert 幂等）。"""
    col = get_collection()
    ec = get_embed_client()
    files = [md for md in NOTES_DIR.rglob("*.md")
             if ".obsidian" not in md.parts]          # 跳过 Obsidian 配置文件夹
    print(f"开始建索引：{len(files)} 个文件")
    total = 0
    for md in files:
        try:
            text = md.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue                                  # 读不了的文件跳过
        pieces = chunk_text(text)
        if not pieces:
            continue
        vectors = embed_texts(pieces, ec)
        rel = md.relative_to(NOTES_DIR)
        col.upsert(                                   # upsert：有则覆盖，无则新增（重复跑不炸）
            ids=[f"{rel}#{i}" for i in range(len(pieces))],
            embeddings=vectors,
            documents=pieces,
            metadatas=[{"file": md.name, "part": i} for i in range(len(pieces))],
        )
        total += len(pieces)
        print(f"  {md.name}: {len(pieces)} 块")
    print(f"索引完成：共 {total} 块，存放在 {CHROMA_DIR}")


# ---------- 第 4 步：语义检索 ----------
def semantic_search(query: str, top_k: int = 5) -> str:
    """按'意思'搜笔记：问题 → 坐标 → 找最近的 top_k 块。"""
    col = get_collection()
    if col.count() == 0:
        return "向量索引还是空的。请先运行：python build_index.py"
    qv = embed_texts([query])[0]
    res = col.query(query_embeddings=[qv], n_results=top_k)
    lines = []
    for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        similarity = 1 - dist                         # cosine 距离 = 1 - 相似度
        lines.append(f"[{meta['file']} 第{meta['part'] + 1}块 相似度{similarity:.2f}]\n{doc[:300]}")
    return "\n\n".join(lines)
