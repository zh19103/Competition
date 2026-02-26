from typing import List, Dict, Any, Optional
import chromadb
import numpy as np
from sentence_transformers import SentenceTransformer
from .config import ChromaConfig, EmbedConfig
from .utils import auto_device

class Retriever:
    def __init__(self, chroma_cfg: ChromaConfig, embed_cfg: EmbedConfig):
        device = embed_cfg.device or auto_device()
        self.model = SentenceTransformer(embed_cfg.embed_model_name, device=device)
        self.embed_cfg = embed_cfg
        self.client = chromadb.PersistentClient(path=chroma_cfg.db_path)
        try:
            self.collection = self.client.get_or_create_collection(
                name=chroma_cfg.collection, metadata={"hnsw:space": chroma_cfg.hnsw_space}
            )
        except TypeError:
            self.collection = self.client.get_or_create_collection(name=chroma_cfg.collection)

    def _encode(self, text: str) -> np.ndarray:
        return self.model.encode(
            [text],
            batch_size=1,
            convert_to_numpy=True,
            normalize_embeddings=self.embed_cfg.normalize,
            show_progress_bar=False
        ).astype("float32")

    def query(self, query: str, top_k: int = 4, include_distances: bool = True):
        q = (query or "").strip()
        if not q:
            return []
        q_emb = self._encode(q)
        res = self.collection.query(
            query_embeddings=q_emb,
            n_results=top_k,
            include=["documents", "ids", "metadatas"] + (["distances"] if include_distances else [])
        )
        docs = res.get("documents", [[]])[0]
        ids_ = res.get("ids", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        dists = res.get("distances", [[]])[0] if include_distances else [None] * len(docs)
        hits = []
        for _id, doc, meta, dist in zip(ids_, docs, metas, dists):
            hits.append({
                "id": _id,
                "document": doc,
                "metadata": meta,
                "distance": float(dist) if dist is not None else None
            })
        return hits
