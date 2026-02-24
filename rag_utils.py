import os
import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer

# 配置项（单独抽出，便于修改）
CONFIG = {
    "csv_path": "test_jobs.csv",
    "db_path": "./a13_vector_db",
    "coll_name": "test_jobs",
    "model_name": "all-MiniLM-L6-v2"
}

# 初始化模型和向量库（全局只初始化一次，提升性能）
model = SentenceTransformer(
    CONFIG["model_name"],
    device="cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") not in (None, "", "-1") else "cpu"
)
db = chromadb.PersistentClient(path=CONFIG["db_path"])
try:
    collection = db.get_or_create_collection(
        name=CONFIG["coll_name"],
        metadata={"hnsw:space": "cosine"}
    )
except TypeError:
    collection = db.get_or_create_collection(name=CONFIG["coll_name"])

def preprocess_data(csv_path):
    """数据预处理：读取CSV→统一列名→填充空值→拼接文本"""
    df = pd.read_csv(csv_path, encoding="utf-8")
    col_map = {"岗位名称": "job", "技能要求": "skills", "薪资范围": "salary", "晋升路径": "path"}
    df = df.rename(columns=col_map)
    for c in ["job", "skills", "salary", "path"]:
        df[c] = df[c].fillna("") if c in df.columns else ""
    df["text"] = (
        "岗位：" + df["job"].astype(str) +
        "，技能要求：" + df["skills"].astype(str) +
        "，薪资：" + df["salary"].astype(str) +
        "，晋升路径：" + df["path"].astype(str)
    )
    return df["text"].tolist(), [f"job_{i}" for i in range(len(df))]

def load_data_to_db(text_chunks, ids):
    """幂等入库：仅添加增量数据"""
    existing = set()
    try:
        peek = collection.peek()
        if peek and "ids" in peek:
            existing = set(peek["ids"])
    except Exception:
        pass
    to_add_idx = [i for i, _id in enumerate(ids) if _id not in existing]
    if not to_add_idx:
        return "无增量数据，无需入库"
    texts_to_add = [text_chunks[i] for i in to_add_idx]
    ids_to_add = [ids[i] for i in to_add_idx]
    embeddings = model.encode(
        texts_to_add,
        batch_size=128,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False
    ).astype("float32")
    metadatas = [{"row_index": int(i)} for i in to_add_idx]
    collection.add(embeddings=embeddings, documents=texts_to_add, ids=ids_to_add, metadatas=metadatas)
    return f"成功入库 {len(to_add_idx)} 条数据"

def search_jobs(query: str, top_k: int = 2, filter_conditions: dict = None):
    """检索岗位：返回(ID, 相似度, 文档, 元数据)"""
    q = (query or "").strip()
    if not q:
        return []
    q_emb = model.encode(
        [q],
        batch_size=1,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False
    ).astype("float32")
    res = collection.query(
        query_embeddings=q_emb,
        n_results=top_k,
        include=["documents", "distances", "ids", "metadatas"],
        where=filter_conditions
    )
    docs = res.get("documents", [[]])[0]
    dists = res.get("distances", [[]])[0]
    ids_ = res.get("ids", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    # 转化为相似度，便于后续匹配
    results = []
    for _id, dist, doc, meta in zip(ids_, dists, docs, metas):
        results.append({
            "id": _id,
            "similarity": round(1 - dist, 4),  # 相似度（0~1）
            "content": doc,
            "metadata": meta
        })
    return results

# 测试入口（可选，开发时验证用）
if __name__ == "__main__":
    # 1. 预处理并入库
    text_chunks, ids = preprocess_data(CONFIG["csv_path"])
    print(load_data_to_db(text_chunks, ids))
    # 2. 检索测试
    query = "我学Python，适合什么岗位？"
    hits = search_jobs(query, top_k=2)
    print("检索结果：")
    for hit in hits:
        print(f"- 相似度：{hit['similarity']} | 内容：{hit['content']}")
