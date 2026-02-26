from typing import Dict, Any, List
from .config import RAGConfig
from .indexer import ExcelIndexer
from .retriever import Retriever
from .generator import QwenGenerator

class RAGPipeline:
    def __init__(self, cfg: RAGConfig = RAGConfig()):
        self.cfg = cfg
        self.indexer = ExcelIndexer(cfg.chroma, cfg.embed)
        self.retriever = Retriever(cfg.chroma, cfg.embed)
        self.generator = QwenGenerator(cfg.qwen)

    def build_index_from_excel(self, excel_path: str) -> int:
        return self.indexer.add_from_excel(excel_path, self.cfg.excel)

    def _format_context(self, hits: List[Dict[str, Any]]) -> str:
        lines = []
        for i, h in enumerate(hits, 1):
            prefix = f"[文档{i}]"
            if h.get("distance") is not None:
                sim = 1 - float(h["distance"])
                prefix += f"(相似度≈{sim:.3f})"
            lines.append(f"{prefix}\n{h['document']}")
        return "\n\n".join(lines) if lines else "（无检索结果）"

    def answer(self, query: str, top_k: int = None) -> Dict[str, Any]:
        k = top_k or self.cfg.top_k
        hits = self.retriever.query(query, top_k=k, include_distances=self.cfg.return_distances)
        user_prompt = self.cfg.user_prompt_template.format(query=query, context=self._format_context(hits))
        answer = self.generator.generate(self.cfg.system_prompt, user_prompt)
        return {"query": query, "answer": answer, "references": hits}
