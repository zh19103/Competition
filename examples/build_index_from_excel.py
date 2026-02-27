# -- coding: utf-8 --
#!/usr/bin/env python3
# -*- coding: ascii -*-
"""
Build a Chroma collection from an Excel file.

- Reads rows from an Excel sheet
- Creates ids/documents/embeddings
- Upserts to a Chroma collection in batches
- Includes metadata to satisfy Chroma validation
"""

import os
import math
import argparse
from typing import List, Dict, Any

import pandas as pd

# You can switch to your own embedding model if you already have one.
# If you prefer to avoid sentence-transformers dependency, replace embed_texts().
try:
    from sentence_transformers import SentenceTransformer
    _HAS_ST = True
except Exception:
    _HAS_ST = False

try:
    import chromadb
    from chromadb.config import Settings
except Exception as e:
    raise RuntimeError("chromadb is required: pip install chromadb") from e


# -------- Configurable defaults --------

# Columns mapping between your Excel and the script.
COL = {
    "id": "岗位编码",       # 可选：用岗位编码当ID，无则自动生成row-xxx
    "title": "岗位名称",    # 你的Excel列名
    "company": "公司名称",  # 你的Excel列名
    "addr": "地址",        # 你的Excel列名
    "date": "更新日期",     # 你的Excel列名
    "text": "岗位详情",    # 必选：用于生成向量的核心文本列（可改成你想拼接的列）
}
# Batch size for upserts
BATCH_SIZE = 128

# Whether to include enhanced business fields in metadata
USE_ENHANCED_METADATA = True

# Chroma DB path and collection name
CHROMA_PERSIST_DIR = os.environ.get("CHROMA_DB_DIR", ".chroma")
CHROMA_COLLECTION = os.environ.get("CHROMA_COLLECTION", "jobs_collection")

# Best-quality Chinese embedding by default; override with EMBED_MODEL if needed.
# Note: larger model -> better quality but slower and more memory usage.
EMBED_MODEL = os.environ.get("EMBED_MODEL", "BAAI/bge-small-zh-v1.5")


# -------- Helpers --------

def _is_nan(v: Any) -> bool:
    # Treat NaN-like values as empty
    try:
        return v is None or (isinstance(v, float) and math.isnan(v))
    except Exception:
        return v is None

def _s(v: Any) -> str:
    # Safe string conversion for metadata/documents
    return "" if _is_nan(v) else str(v).strip()

def _normalize_text_zh(s: str) -> str:
    # Basic Chinese text normalization: remove special spaces and collapse whitespace
    if s is None:
        return ""
    s = str(s)
    s = s.replace("\u3000", " ").replace("\xa0", " ")
    return " ".join(s.split())

def _maybe_add_prefix_for_model(texts: List[str], is_query: bool) -> List[str]:
    # For e5-family embeddings, uncomment below to add prefixes
    # name = EMBED_MODEL.lower()
    # if "e5" in name:
        # prefix = "query: " if is_query else "passage: "
        # return [prefix + t for t in texts]
    return texts

def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Produce embeddings for given texts (Chinese-friendly).
    Replace this with your own embedding pipeline if needed.
    """
    if not texts:
        return []
    texts = [_normalize_text_zh(t) for t in texts]
    texts = _maybe_add_prefix_for_model(texts, is_query=False)
    if _HAS_ST:
        model = SentenceTransformer(EMBED_MODEL)
        # normalize_embeddings=True keeps cosine-sim comparable to query side
        embs = model.encode(texts, normalize_embeddings=True, show_progress_bar=False, batch_size=64)
        return [list(map(float, row)) for row in embs]
    # Fallback (not for production)
    def _hash_vec(t: str, dim: int = 1024) -> List[float]:
        import hashlib
        h = hashlib.sha256(t.encode("utf-8")).digest()
        b = (h * ((dim + len(h) - 1) // len(h)))[:dim]
        return [((x / 255.0) - 0.5) for x in b]
    return [_hash_vec(t) for t in texts]


# -------- Core pipeline --------

def load_excel(path: str, sheet: str = None) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Excel file not found: {path}")
    if sheet is None:
        sheet = "Sheet1"
    try:
        df = pd.read_excel(path, sheet_name=sheet, engine="openpyxl")
    except Exception:
        df = pd.read_excel(path, sheet_name=sheet, engine="xlrd")
    if isinstance(df, dict):
        raise ValueError(f"Excel文件包含多个工作表，请用--sheet指定具体工作表名！\n可用工作表：{list(df.keys())}")
    df.columns = [str(c).strip() for c in df.columns]
    return df


def build_rows(df: pd.DataFrame) -> Dict[str, List[Any]]:
    text_col = COL["text"]
    if text_col not in df.columns:
        raise ValueError(f"Required text column not found: {text_col!r}. "
                         f"Available columns: {list(df.columns)}")

    ids = []
    docs = []

    has_id_col = COL["id"] in df.columns
    for i in range(len(df)):
        if has_id_col:
            rid = _s(df.loc[i, COL["id"]])
            if not rid:
                rid = f"row-{i+1}"
        else:
            rid = f"row-{i+1}"
        raw_doc = _s(df.loc[i, text_col])
        doc = _normalize_text_zh(raw_doc)
        ids.append(rid)
        docs.append(doc)

    return {"ids": ids, "docs": docs}

def build_metas_batch(df: pd.DataFrame, row_indices: List[int], ids_b: List[str]) -> List[Dict[str, Any]]:
    if not USE_ENHANCED_METADATA:
        return [{"source": "excel"} for _ in ids_b]

    metas = []
    for idx_in_batch, rid in enumerate(ids_b):
        row_idx = row_indices[idx_in_batch]
        meta = {
            "id": rid,
            "title": _normalize_text_zh(_s(df.loc[row_idx, COL["title"]])) if COL["title"] in df.columns else "",
            "company": _normalize_text_zh(_s(df.loc[row_idx, COL["company"]])) if COL["company"] in df.columns else "",
            "addr": _normalize_text_zh(_s(df.loc[row_idx, COL["addr"]])) if COL["addr"] in df.columns else "",
            "date": _normalize_text_zh(_s(df.loc[row_idx, COL["date"]])) if COL["date"] in df.columns else "",
            "source": "excel",
        }
        metas.append(meta)
    return metas

def ensure_collection(client, name: str):
    try:
        col = client.get_collection(name=name)
    except Exception:
        col = client.create_collection(name=name)
    return col

def main():
    parser = argparse.ArgumentParser(description="Build Chroma collection from Excel.")
    parser.add_argument("--excel", required=True, help="Path to .xlsx file")
    parser.add_argument("--sheet", default=None, help="Excel sheet name (optional)")
    parser.add_argument("--persist", default=CHROMA_PERSIST_DIR, help="Chroma persist directory")
    parser.add_argument("--collection", default=CHROMA_COLLECTION, help="Chroma collection name")
    parser.add_argument("--batch", type=int, default=BATCH_SIZE, help="Upsert batch size")
    args = parser.parse_args()

    # Load data
    df = load_excel(args.excel, args.sheet)
    rows = build_rows(df)
    ids_all = rows["ids"]
    docs_all = rows["docs"]

    # Deduplicate IDs (make unique by suffix if repeated)
    unique_ids = []
    unique_docs = []
    seen_ids = set()

    for rid, doc in zip(ids_all, docs_all):
        if rid not in seen_ids:
            seen_ids.add(rid)
            unique_ids.append(rid)
            unique_docs.append(doc)
        else:
            suffix = 1
            while f"{rid}_{suffix}" in seen_ids:
                suffix += 1
            new_rid = f"{rid}_{suffix}"
            seen_ids.add(new_rid)
            unique_ids.append(new_rid)
            unique_docs.append(doc)

    ids_all = unique_ids
    docs_all = unique_docs
    print(f"IDs deduped: {len(rows['ids'])} -> {len(ids_all)}")

    # Build embeddings
    embs_all = embed_texts(docs_all)

    # Init Chroma client
    client = chromadb.PersistentClient(
        path=args.persist,
        settings=Settings(allow_reset=False)
    )
    col = ensure_collection(client, args.collection)

    # Upsert in batches
    n = len(ids_all)
    bs = max(1, int(args.batch))
    for s in range(0, n, bs):
        e = min(s + bs, n)
        ids_b = ids_all[s:e]
        docs_b = docs_all[s:e]
        embs_b = embs_all[s:e]
        row_indices = list(range(s, e))
        metas_b = build_metas_batch(df, row_indices, ids_b)

        col.upsert(
            ids=ids_b,
            documents=docs_b,
            embeddings=embs_b,
            metadatas=metas_b,
        )

    print(f"Upserted {n} rows into collection '{args.collection}' at '{args.persist}'")
    print(f"Embedding model: {EMBED_MODEL}")
    print("Done.")

if __name__ == "__main__":
    main()
