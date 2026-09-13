"""
build_index.py —— 建向量索引（阶段 2）
用法：python build_index.py   （跑一次即可；笔记大改后重跑也安全，upsert 不会重复）
"""
from dotenv import load_dotenv

load_dotenv()
from rag_lib import build_index  # noqa: E402

if __name__ == "__main__":
    build_index()
