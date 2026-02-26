from typing import List, Dict, Any, Optional
import numpy as np
import chromadb
from sentence_transformers import SentenceTransformer
from .config import ChromaConfig, EmbedConfig, ExcelIngestConfig
from .utils import auto_device, load_excel, normalize_frame, build_texts_from_df

class ExcelIndexer:
    def __init__(self, chroma_cfg: ChromaConfig, embed_cfg: EmbedConfig):
        device = embed_cfg.device or auto_device()
        self.model = SentenceTransformer(embed_cfg.embed_model_name, device=device)
        self.embed_cfg = embed_cfg
        self.client = chromadb.PersistentClient(path=chroma_cfg.db_path)
        self.collection = self._get_collection(chroma_cfg)

    def _get_collection(self, cfg: ChromaConfig):
        try:
            return self.client.get_or_create_collection(
                name=cfg.collection, metadata={"hnsw:space": cfg.hnsw_space}
            )
        except TypeError:
            return self.client.get_or_create_collection(name=cfg.collection)

    def _encode(self, texts: List[str]) -> np.ndarray:
        emb = self.model.encode(
            texts,
            batch_size=self.embed_cfg.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=self.embed_cfg.normalize,
            show_progress_bar=False
        ).astype("float32")
        return emb

    def add_texts(
        self,
        texts: List[str],
        ids: Optional[List[str]] = None,
        metadatas: Optional[List[Dict[str, Any]]] = None,
        skip_existing: bool = True
    ) -> int:
        n = len(texts)
        if ids is None:
            ids = [f"doc_{i}" for i in range(n)]

        to_add_idx = list(range(n))
        if skip_existing:
            existing = set()
            try:
                peek = self.collection.peek()
                if peek and "ids" in peek:
                    existing = set(peek["ids"])
            except Exception:
                pass
            to_add_idx = [i for i in to_add_idx if ids[i] not in existing]
        if not to_add_idx:
            return 0

        batch_texts = [texts[i] for i in to_add_idx]
        batch_ids = [ids[i] for i in to_add_idx]
        batch_metas = [metadatas[i] for i in to_add_idx] if metadatas else None

        embeddings = self._encode(batch_texts)
        self.collection.add(
            embeddings=embeddings,
            documents=batch_texts,
            ids=batch_ids,
            metadatas=batch_metas
        )
        return len(batch_ids)

    def add_from_excel(self, excel_path: str, ingest_cfg: ExcelIngestConfig) -> int:
        df = load_excel(excel_path, sheet_name=ingest_cfg.sheet_name)
        df_norm = normalize_frame(df, column_map=ingest_cfg.column_map or {})
        texts = build_texts_from_df(df_norm, ingest_cfg.text_template)
        # 如果源表带“岗位编码”，优先用作 ID；否则用前缀+行号
        ids = (
            df["岗位编码"].fillna("").astype(str).tolist()
            if "岗位编码" in df.columns and df["岗位编码"].notna().any()
            else [f"{ingest_cfg.id_prefix}{i}" for i in range(len(texts))]
        )
        # 元数据保留关键字段，便于前端展示/过滤
        metas = []
        for i in range(len(texts)):
            meta = {"row_index": int(i)}
            for src, dst in (ingest_cfg.column_map or {}).items():
                if src in df.columns:
                    meta[dst] = str(df.iloc[i][src]) if df.iloc[i][src] is not None else ""
            metas.append(meta)
        return self.add_texts(texts, ids=ids, metadatas=metas, skip_existing=True)
