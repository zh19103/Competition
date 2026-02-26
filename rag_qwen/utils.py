import os
from typing import Optional, Dict, List
import pandas as pd

def auto_device(env_key: str = "CUDA_VISIBLE_DEVICES") -> str:
    visible = os.environ.get(env_key)
    return "cpu" if visible in (None, "", "-1") else "cuda"

def load_excel(path: str, sheet_name: Optional[str] = None) -> pd.DataFrame:
    # 需安装 openpyxl（.xlsx）或 xlrd(旧 .xls)
    return pd.read_excel(path, sheet_name=sheet_name)

def normalize_frame(df: pd.DataFrame, column_map: Dict[str, str]) -> pd.DataFrame:
    logical_cols = list(dict.fromkeys(column_map.values()))
    out = {}
    for src_col, dst_col in column_map.items():
        out[dst_col] = df[src_col].fillna("").astype(str) if src_col in df.columns else ""
    return pd.DataFrame(out)[logical_cols]

def build_texts_from_df(df: pd.DataFrame, text_template: str) -> List[str]:
    records = df.to_dict(orient="records")
    texts = []
    for rec in records:
        safe = {k: (str(v) if v is not None else "") for k, v in rec.items()}
        texts.append(text_template.format(**safe))
    return texts
