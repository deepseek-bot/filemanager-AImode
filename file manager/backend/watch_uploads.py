import os, time, logging, uuid
import requests
import chromadb
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from chromadb import Client
from chromadb.config import Settings

# 从你的 main.py 导入解析与切块函数
from main import read_text_from_file, chunk_text

UPLOAD_DIR = "/data/uploads"
QUEUE_DIR = "/data/queue"
ERROR_DIR = "/data/error"
CHROMA_DIR = "/data/chroma"

# 你的切块参数，如果在 main.py 里定义了常量，也可以一并导入
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(QUEUE_DIR, exist_ok=True)
os.makedirs(ERROR_DIR, exist_ok=True)
os.makedirs(CHROMA_DIR, exist_ok=True)

client = chromadb.PersistentClient(path="/data/chroma")
collection = client.get_or_create_collection("alldata")

def wait_for_complete(filepath, wait_time=1, timeout=30):
    prev_size = -1
    start = time.time()
    while time.time() - start < timeout:
        try:
            size = os.path.getsize(filepath)
        except FileNotFoundError:
            return False
        if size == prev_size:
            return True
        prev_size = size
        time.sleep(wait_time)
    return False

def should_ignore(filename):
    # 保留你原先的忽略策略；不再限制 .txt
    if filename.startswith(".") or filename.endswith(".tmp") or filename.startswith("~"):
        return True
    if "__" in filename:
        return True
    return False

class Handler(FileSystemEventHandler):
    processing = set()

    def on_created(self, event):
        if event.is_directory:
            return
        dirname, filename = os.path.split(event.src_path)
        if should_ignore(filename):
            return
        if event.src_path in self.processing:
            return
        self.processing.add(event.src_path)
        try:
            if not wait_for_complete(event.src_path):
                logging.warning(f"文件未稳定，跳过: {filename}")
                return
            new_name = f"{uuid.uuid4()}__{filename}"
            queued_path = os.path.join(QUEUE_DIR, new_name)
            os.rename(event.src_path, queued_path)
            logging.info(f"已入队: {new_name}")
            self.process_file(queued_path)
        finally:
            self.processing.discard(event.src_path)

    def process_file(self, filepath):
        filename = os.path.basename(filepath)
        try:
            logging.info(f"开始处理: {filename}")
            # 用你 main.py 的解析器（支持 .txt/.docx/.pdf）
            text = read_text_from_file(filepath)
            chunks = chunk_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
            if not chunks:
                raise ValueError("解析后文本为空或未生成任何 chunk")

            # 调用 Ollama 生成每个 chunk 的嵌入
            embeddings = []
            for i, chunk in enumerate(chunks):
                resp = requests.post(
                    "http://ollama:11434/api/embeddings",
                    json={"model": "nomic-embed-text", "prompt": chunk},
                    timeout=60
                )
                resp.raise_for_status()
                data = resp.json()
                emb = data.get("embedding") or (data.get("embeddings")[0] if "embeddings" in data else None)
                if not isinstance(emb, list) or len(emb) == 0:
                    raise ValueError(f"chunk {i} 未获得有效 embedding")
                embeddings.append(emb)

            # 批量入库（每个 chunk 一个 id，保留原文件名作为前缀）
            ids = [f"{filename}__chunk_{i}" for i in range(len(chunks))]
            collection.add(
                documents=chunks,
                embeddings=embeddings,
                ids=ids,
                metadatas=[{"source": filename, "chunk_index": i} for i in range(len(chunks))]
            )
            client.persist()

            logging.info(f"入库成功: {filename} | 写入 {len(chunks)} 个 chunk | 当前总条数: {collection.count()}")

            # 可选：进行一次查询验证
            try:
                q = collection.query(query_texts=[chunks[0][:200]], n_results=1)
                logging.info(f"查询验证返回 IDs: {q.get('ids')}")
            except Exception as qe:
                logging.warning(f"查询验证失败: {qe}")

            os.remove(filepath)

        except Exception as e:
            logging.error(f"处理失败 {filename}: {e}")
            os.rename(filepath, os.path.join(ERROR_DIR, filename))

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    observer = Observer()
    observer.schedule(Handler(), UPLOAD_DIR, recursive=False)
    observer.start()
    logging.info(f"正在监控文件夹: {UPLOAD_DIR}")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logging.info("收到退出信号，正在停止监听...")
        observer.stop()
    observer.join()

if __name__ == "__main__":
    main()
