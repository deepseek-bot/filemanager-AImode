
# main.py
import os
import uuid
import shutil
from typing import Optional, List
from collections import defaultdict

import requests
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi import Query
from tqdm import tqdm

import chromadb
from chromadb.config import Settings

import PyPDF2
from docx import Document
import logging


# ---------- 配置（可用环境变量覆盖） ----------
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://ollama:11434")
EMBED_MODEL = os.getenv("EMBED_MODEL", "qwen2.5:3b")
GENERATE_MODEL = os.getenv("GENERATE_MODEL", "qwen2.5:3b")
DATA_DIR = os.getenv("DATA_DIR", "/data")
CHROMA_DIR = os.getenv("CHROMA_DIR", "/data/chroma")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "docs")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1200"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
ALLOW_ORIGINS = os.getenv("ALLOW_ORIGINS", "*")
# -------------------------------------------------

os.makedirs(os.path.join(DATA_DIR, "uploads"), exist_ok=True)
os.makedirs(CHROMA_DIR, exist_ok=True)

app = FastAPI(title="Personal RAG Backend")
chat_history = defaultdict(list)

# CORS（开发方便，生产请限制 origin）
app.add_middleware(
    CORSMiddleware,
    allow_origins=[ALLOW_ORIGINS] if ALLOW_ORIGINS != "*" else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



# 用 PersistentClient，而不是 Client+Settings
chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)

def read_text_from_file(path: str) -> str:
    path = str(path)
    ext = os.path.splitext(path)[1].lower()

    if ext == ".txt":
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()

    elif ext == ".docx":
        try:
            doc = Document(path)
            return "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
        except Exception as e:
            raise ValueError(f".docx 文件解析失败(File parse error): {e}")

    elif ext == ".pdf":
        text_parts = []
        with open(path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    text_parts.append(text)
        return "\n".join(text_parts)

    else:
        raise ValueError(f"不支持的文件类型（files not supported）: {ext}")

def chunk_text(text: str, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP) -> List[str]:
    if not text:
        return []
    chunks = []
    start = 0
    L = len(text)
    while start < L:
        end = min(start + size, L)
        chunks.append(text[start:end])
        start = end - overlap
        if start < 0:
            start = 0
    return chunks

def parse_embedding_response(resp_json):
    # Ollama 的 embedding 返回结构可能有不同，做稳健解析
    if isinstance(resp_json, dict):
        if "embedding" in resp_json:
            return resp_json["embedding"]
        if "data" in resp_json and isinstance(resp_json["data"], list):
            first = resp_json["data"][0]
            if isinstance(first, dict) and "embedding" in first:
                return first["embedding"]
    # 若不匹配，直接 error
    raise ValueError("无法解析 embedding 返回：" + str(resp_json))

def get_embedding(text: str):
    url = f"{OLLAMA_URL.rstrip('/')}/api/embeddings"
    payload = {"model": EMBED_MODEL, "input": text}
    r = requests.post(url, json=payload, timeout=60)
    r.raise_for_status()
    emb = parse_embedding_response(r.json())
    # 格式校验：必须是非空的一维 float 列表
    if(
        not isinstance(emb, list)
        or len(emb) == 0
        or not all(isinstance(x, (float, int)) for x in emb)
    ):
        logging.error("get_embedding 返回格式不合法: %r", emb)
        raise ValueError("无效的 embedding 格式")
    return emb

def parse_generate_response(resp_json):
    # Ollama generate 常见返回 response 字段
    if isinstance(resp_json, dict):
        if "response" in resp_json:
            return resp_json["response"]
        if "text" in resp_json:
            return resp_json["text"]
        if "choices" in resp_json and isinstance(resp_json["choices"], list):
            return resp_json["choices"][0].get("message") or resp_json["choices"][0].get("text")
    return str(resp_json)

def generate_answer(prompt: str):
    """
    调用本地 Ollama 模型生成回答
    """
    url = f"{OLLAMA_URL.rstrip('/')}/api/generate"
    payload = {"model": GENERATE_MODEL, "prompt": prompt}
    r = requests.post(url, json=payload, timeout=120)
    r.raise_for_status()
    return parse_generate_response(r.json())


# =========================================================
# 文件上传与向量生成模块
# =========================================================
@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    """
    上传文件 -> 读取内容 -> 分chunk -> 生成embedding -> 写入数据库
    """
    filename = file.filename
    uid = str(uuid.uuid4())
    save_path = os.path.join(DATA_DIR, "uploads", f"{uid}__{filename}")
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    try:
        text = read_text_from_file(save_path)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"文件解析失败: {e}")
    if not text.strip():
        return {"status": "empty", "msg": "解析后无文本内容", "filename": filename}

    chunks = chunk_text(text)
    ids, docs, metas, embs = [], [], [], []

    # 批量逐 chunk 生成 embedding
    for i, c in enumerate(chunks):
        try:
            emb = get_embedding(c)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"向量生成失败（embedding error）: {e}")
        ids.append(f"{uid}_{i}")
        docs.append(c)
        metas.append({"source": filename, "chunk_index": i})
        embs.append(emb)
    # ========校验 embs 列表========
    for emb in embs:
        if (
            not isinstance(emb, list)
            or len(emb) == 0
            or not all(isinstance(x, (float, int)) for x in emb)
        ):
            logging.error("upload:单个 chunk embedding 无效: %r", emb)
            raise HTTPException(status_code=500, detail="内部错误：无效的块向量格式")
    
    collection.add(ids=ids, documents=docs, metadatas=metas, embeddings=embs)
    
    try:
        chroma_client.persist()
    except Exception as e:
        logging.warning(f"Chroma persist 失败: {e}")

    return {"status": "ok", "file": filename, "chunks": len(chunks)}


# =========================================================
# 查询逻辑模块
# =========================================================
def process_query(query: str, session_id: str):
    """
    查询主逻辑模块（可被 /api/ask 调用）
    """
    if not query:
        raise HTTPException(status_code=400, detail="缺少查询参数 (Missing query parameter)")

    chat_history[session_id].append(("user", query))

    try:
        qemb = get_embedding(query)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"查询向量生成失败 (Query embedding error)：{e}")

    #格式校验与包装
    if not isinstance(qemb, list) or len(qemb) == 0:
        logging.error("process_query:查询 embedding 无效: %r", qemb)
        raise HTTPException(status_code=500, detail="内部错误：空查询向量 (Invalid query embedding format)")
    query_embs = [qemb]   # 确保二维列表

    try:
        res=collection.query(
            query_embeddings=query_embs,
            n_results=3,
            include=["documents", "metadatas", "distances"]
        )
    except Exception as e:
         logging.exception("Chroma query 失败")
         raise HTTPException(status_code=500, detail=f"检索失败（retrieval error）: {e}")

    docs_list = res.get("documents")
    if not docs_list or not docs_list[0]:  # 没有检索到文档
        ans = "数据库里还没有内容，请先上传文档再来问问题。"
        chat_history[session_id].append(("assistant", ans))
        return {
            "answer": ans,
            "retrieved": [],
            "session_id": session_id,
            "history": chat_history[session_id][-6:]
        }
    else:
         docs = docs_list[0]

    ctx = "\n\n".join(docs)
    history_text = "\n".join([f"{role.upper()}: {msg}" for role, msg in chat_history[session_id][-5:]])

    prompt = f"""已知内容：
{ctx}

对话历史：
{history_text}

现在用户问：{query}
请基于已知内容和历史对话回答。找不到就说明找不到。
"""

    try:
        ans = generate_answer(prompt)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"生成失败（answer generation error）: {e}")

    chat_history[session_id].append(("assistant", ans))

    return {
        "answer": ans,
        "retrieved": docs,
        "session_id": session_id,
        "history": chat_history[session_id][-6:]
    }


# =========================================================
# 路由层
# =========================================================
@app.post("/api/ask")
async def ask(query: Optional[str] = Form(None), session_id: str = Form("default")):
    """
    主问答接口
    """
    return process_query(query, session_id)


@app.post("/api/clear_session")
async def clear_session(session_id: str = Form("default")):
    """
    清空指定会话
    """
    if session_id in chat_history:
        chat_history[session_id] = []
        return {"status": "ok", "msg": f"会话 {session_id} 已清空"}
    return {"status": "not_found", "msg": f"会话 {session_id} 不存在"}


@app.get("/api/list_sessions")
async def list_sessions():
    """
    列出所有活跃会话
    """
    return {"sessions": list(chat_history.keys())}


@app.get("/api/status")
def status():
    """
    检查服务运行状态
    """
    return {
        "ollama_url": OLLAMA_URL,
        "embed_model": EMBED_MODEL,
        "generate_model": GENERATE_MODEL,
        "chroma_dir": CHROMA_DIR,
        "collection": COLLECTION_NAME
    }
