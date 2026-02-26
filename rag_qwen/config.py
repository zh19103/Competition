from dataclasses import dataclass
from typing import Optional, Literal, Dict

@dataclass
class ChromaConfig:
    db_path: str = "./a13_vector_db"
    collection: str = "job_docs"
    hnsw_space: Literal["cosine", "l2", "ip"] = "cosine"

@dataclass
class EmbedConfig:
    embed_model_name: str = "all-MiniLM-L6-v2"
    device: Optional[str] = None
    batch_size: int = 128
    normalize: bool = True

@dataclass
class QwenConfig:
    use_openai: bool = True
    openai_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    openai_api_key_env: str = "QWEN_API_KEY"
    openai_model: str = "qwen-3.5-plus"
    dashscope_api_key_env: str = "DASHSCOPE_API_KEY"
    dashscope_model: str = "qwen-3.5-plus"
    temperature: float = 0.2
    top_p: float = 0.9
    max_tokens: int = 1024

@dataclass
class ExcelIngestConfig:
    sheet_name: Optional[str] = None  # None=第一个工作表
    # 将你的 Excel 列名映射为逻辑字段
    column_map: Dict[str, str] = None
    # 定制文本模板（已覆盖你的字段）
    text_template: str = (
        "岗位：{title}｜编码：{code}\n"
        "公司：{company}｜行业：{industry}｜规模：{size}｜类型：{ctype}\n"
        "地址：{addr}｜薪资：{salary}｜更新日期：{date}\n"
        "岗位详情：{job_detail}\n"
        "公司详情：{company_detail}"
    )
    id_prefix: str = "job_"

@dataclass
class RAGConfig:
    chroma: ChromaConfig = ChromaConfig()
    embed: EmbedConfig = EmbedConfig()
    qwen: QwenConfig = QwenConfig()
    excel: ExcelIngestConfig = ExcelIngestConfig(
        column_map={
            "岗位名称": "title",
            "地址": "addr",
            "薪资范围": "salary",
            "公司名称": "company",
            "所属行业": "industry",
            "公司规模": "size",
            "公司类型": "ctype",
            "岗位编码": "code",
            "岗位详情": "job_detail",
            "更新日期": "date",
            "公司详情": "company_detail",
        }
    )
    top_k: int = 4
    return_distances: bool = True
    system_prompt: str = (
        "你是一名招聘与求职咨询助手。请仅依据提供的参考资料回答，避免编造；"
        "给出明确建议与可执行步骤，如资料不足请说明并给出下一步建议。"
    )
    user_prompt_template: str = (
        "用户问题：{query}\n\n"
        "参考资料（仅可依据此处内容作答）：\n"
        "{context}\n\n"
        "请先输出结论，再给依据（引用资料中的关键短语/字段），必要时列出备选岗位或公司。"
    )

