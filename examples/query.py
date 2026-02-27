#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import os
import sys
import json
import argparse
from typing import List, Dict, Any, Optional

import chromadb
from sentence_transformers import SentenceTransformer

CHROMA_PERSIST_DIR = ".chroma"
COLLECTION_NAME = "jobs_collection"
EMBED_MODEL = "BAAI/bge-small-zh-v1.5"

_MODEL_CACHE: Optional[SentenceTransformer] = None
_COLLECTION_CACHE = None


def init_chroma():
    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    try:
        col = client.get_collection(name=COLLECTION_NAME)
    except Exception as e:
        raise RuntimeError(
            f"找不到集合 {COLLECTION_NAME}，请先运行构建脚本生成向量库"
        ) from e
    return col


def init_model():
    try:
        model = SentenceTransformer(EMBED_MODEL)
        return model
    except Exception as e:
        raise RuntimeError(
            "模型加载失败，请确认已安装 sentence-transformers：pip install sentence-transformers"
        ) from e


def get_collection_cached():
    global _COLLECTION_CACHE
    if _COLLECTION_CACHE is None:
        _COLLECTION_CACHE = init_chroma()
    return _COLLECTION_CACHE


def get_model_cached():
    global _MODEL_CACHE
    if _MODEL_CACHE is None:
        _MODEL_CACHE = init_model()
    return _MODEL_CACHE


def _to_similarity_from_cosine_distance(d: Any) -> float:
    # Chroma 的 distances 为 cosine distance（0~2），这里映射到相似度（0~1）
    try:
        sim = 1.0 - float(d) / 2.0
    except Exception:
        return 0.0
    return max(0.0, min(1.0, sim))


def _ensure_2d_embedding(emb) -> List[List[float]]:
    if hasattr(emb, "tolist"):
        emb = emb.tolist()
    if isinstance(emb, (list, tuple)) and emb and isinstance(emb[0], (int, float)):
        return [list(map(float, emb))]
    if isinstance(emb, (list, tuple)) and emb and isinstance(emb[0], (list, tuple)):
        return [list(map(float, emb[0]))]
    raise RuntimeError("模型返回的 embedding 形态异常")


def _pick_first_nonempty(meta: Dict[str, Any], keys: List[str], default: str = "未知") -> str:
    for k in keys:
        v = meta.get(k)
        if v is None:
            continue
        s = str(v).strip()
        if s:
            return s
    return default


def _normalize_qwen_model_name(name: str) -> str:
    """
    纠正常见的模型名写法，避免 URL 拼接失败。
    """
    if not name:
        return "qwen-plus"
    n = name.strip().lower()
    aliases = {
        "qwen3.5-plus": "qwen-3.5-plus",
        "qwen-3_5-plus": "qwen-3.5-plus",
        "qwen3_5-plus": "qwen-3.5-plus",
        "qwen-3.5plus": "qwen-3.5-plus",
        "qwen35-plus": "qwen-3.5-plus",
        "qwen-plus-3.5": "qwen-3.5-plus",
        "qwen3.5": "qwen-3.5",
        "qwen3_5": "qwen-3.5",
        "qwen3-5": "qwen-3.5",
        "qwenplus": "qwen-plus",
        "qwen_plus": "qwen-plus",
        "qwen turbo": "qwen-turbo",
        "qwen-turbo": "qwen-turbo",
        "qwenplus-": "qwen-plus",
        "qwen3.5plus": "qwen-3.5-plus",
        "qwen3_5plus": "qwen-3.5-plus",
    }
    return aliases.get(n, n)


def search_jobs(
    query: str,
    top_k: int = 0,
    where: Optional[Dict[str, Any]] = None,
    where_document: Optional[Dict[str, Any]] = None,
    include: Optional[List[str]] = None,
    use_cache: bool = True,
    min_sim: float = 0.5,
    unlimited: bool = False,
) -> List[Dict[str, Any]]:
    """
    检索岗位：
    - 当 unlimited=True 或 top_k<=0 时，返回“全部相关”：按相似度阈值 min_sim 过滤，数量不限（受 MAX_LIMIT 保护）。
    - 否则返回前 top_k 条（同时也会受 min_sim 过滤）。
    """
    if not isinstance(query, str):
        raise TypeError("query 必须是字符串")
    query = query.strip()
    if not query:
        return []
    if not isinstance(top_k, int):
        raise TypeError("top_k 必须是整数")

    col = get_collection_cached() if use_cache else init_chroma()
    model = get_model_cached() if use_cache else init_model()

    try:
        query_emb = model.encode(query, normalize_embeddings=True)
    except Exception as e:
        raise RuntimeError(f"查询向量生成失败: {e}")
    query_emb = _ensure_2d_embedding(query_emb)

    if include is None:
        include = ["documents", "metadatas", "distances"]
    else:
        include = list(dict.fromkeys(include))

    # 计算 n_results：全量相关模式下使用较大上限；否则按 top_k
    MAX_LIMIT = 5000  # 安全上限，可按库大小调整
    if unlimited or top_k <= 0:
        n_results = MAX_LIMIT
    else:
        n_results = min(max(1, top_k), MAX_LIMIT)

    try:
        results = col.query(
            query_embeddings=query_emb,
            n_results=n_results,
            include=include,
            where=where,
            where_document=where_document,
        )
    except Exception as e:
        raise RuntimeError(f"Chroma 检索失败: {e}")

    docs_batches = results.get("documents") or [[]]
    metas_batches = results.get("metadatas") or [[]]
    dists_batches = results.get("distances") or [[]]

    docs = docs_batches[0] if docs_batches else []
    metas = metas_batches[0] if metas_batches else []
    dists = dists_batches[0] if dists_batches else []

    # 先构造全部，再按相似度阈值和 top_k 截断
    all_items: List[Dict[str, Any]] = []
    N = min(len(docs), len(metas), len(dists))
    for i in range(N):
        dist = dists[i] if i < len(dists) else 2.0
        sim = _to_similarity_from_cosine_distance(dist)
        all_items.append(
            {
                "排名": i + 1,
                "相似度": round(sim, 4),
                "岗位信息": docs[i] if isinstance(docs[i], str) else ("" if docs[i] is None else str(docs[i])),
                "元数据": metas[i] if isinstance(metas[i], dict) else {},
                "原始距离": float(dist) if isinstance(dist, (int, float)) else None,
            }
        )

    # 过滤阈值
    if min_sim < 0.0:
        min_sim = 0.0
    if min_sim > 1.0:
        min_sim = 1.0
    filtered = [x for x in all_items if x.get("相似度", 0.0) >= min_sim]

    # 若非 unlimited 且 top_k>0，再做截断；否则返回全部相关
    if not (unlimited or top_k <= 0):
        filtered = filtered[:top_k]

    # 重排排名
    for idx, item in enumerate(filtered, 1):
        item["排名"] = idx

    return filtered


def call_qwen_with_search_results(
    query: str,
    results: List[Dict[str, Any]],
    model_name: str = "qwen-plus",  # 默认使用 qwen-plus
    temperature: float = 0.2,
    top_p: float = 0.9,
    timeout: int = 60,
) -> str:
    try:
        import dashscope
        from dashscope import Generation
    except ImportError:
        return "请先安装千问依赖：pip install dashscope"

    api_key = os.environ.get("DASHSCOPE_API_KEY")
    if not api_key:
        return "请先设置环境变量 DASHSCOPE_API_KEY（阿里云灵积平台获取）"
    dashscope.api_key = api_key

    # 可选：允许通过环境变量覆盖 base_url（如私有化网关/地域域名）
    base_url = os.environ.get("DASHSCOPE_BASE_URL", "").strip()
    if base_url:
        try:
            dashscope.base_url = base_url
        except Exception:
            pass

    model_name = _normalize_qwen_model_name(model_name)

    if not results:
        return "未找到匹配的岗位。"

    # 组织检索结果上下文，优先抽取：岗位名称、公司、地点、薪资；并附加少量核心描述
    lines = []
    for res in results:
        meta = res.get("元数据", {}) or {}
        title = _pick_first_nonempty(meta, ["title", "job_title"], "未知")
        company = _pick_first_nonempty(meta, ["company", "org", "employer"], "未知")
        location = _pick_first_nonempty(meta, ["addr", "location", "city"], "未知")
        salary = _pick_first_nonempty(meta, ["salary", "pay", "compensation", "salary_range"], "未知")
        sim = res.get("相似度", 0.0)
        text = (res.get("岗位信息", "") or "").replace("\n", " ").strip()
        if len(text) > 160:
            text = text[:160] + "..."
        lines.append(
            f"- 岗位名称：{title}\n"
            f"  公司：{company}\n"
            f"  地点：{location}\n"
            f"  薪资：{salary}\n"
            f"  核心信息：{text}\n"
            f"  相似度：{sim}"
        )
    prompt_context = "\n".join(lines)

    # 强化版系统提示词：强调与用户需求的强关联与展示顺序
    system_prompt_tpl = (
        "你是专业的岗位检索分析助手，需严格按以下要求回答：\n"
        "1. 开头先总结：找到X个符合「{用户需求}」的岗位；\n"
        "2. 分点列出岗位，优先展示「岗位名称、公司、地点、薪资」，再补充核心信息；\n"
        "3. 语言简洁、口语化，适配非技术用户阅读；\n"
        "4. 只返回回答内容，不添加多余解释；\n"
        "5. 重点关联用户查询的核心需求（如薪资、城市、岗位类型）。"
    )
    sys_prompt_final = system_prompt_tpl.replace("X", str(len(results))).replace("{用户需求}", query)
    user_prompt = f"用户需求：{query}\n检索结果（按相似度排序）：\n{prompt_context}"

    # 优先尝试新版 Chat Responses 接口；失败则回退到 Generation.call
    try:
        messages = [
            {"role": "system", "content": sys_prompt_final},
            {"role": "user", "content": user_prompt},
        ]
        used_new_api = False
        resp = None
        try:
            # 优先尝试 Chat 或 Responses（不同 dashscope 版本类名不同）
            if hasattr(dashscope, "Chat") and hasattr(dashscope.Chat, "complete"):
                resp = dashscope.Chat.complete(
                    model=model_name,
                    messages=messages,
                    temperature=float(temperature),
                    top_p=float(top_p),
                    timeout=timeout,
                )
                used_new_api = True
            elif hasattr(dashscope, "Responses") and hasattr(dashscope.Responses, "create"):
                resp = dashscope.Responses.create(
                    model=model_name,
                    input={"messages": messages},
                    parameters={"temperature": float(temperature), "top_p": float(top_p)},
                    timeout=timeout,
                )
                used_new_api = True
        except Exception:
            used_new_api = False
            resp = None

        if used_new_api and resp is not None:
            # 兼容不同返回结构
            text = None
            if isinstance(resp, dict):
                out = resp.get("output") or {}
                choices = out.get("choices") or []
                if choices:
                    msg = choices[0].get("message") or {}
                    content = msg.get("content")
                    if isinstance(content, list) and content and isinstance(content[0], dict):
                        text = content[0].get("text")
                    elif isinstance(content, str):
                        text = content
                if not text:
                    text = out.get("text")
            else:
                if hasattr(resp, "output_text"):
                    text = resp.output_text
                elif hasattr(resp, "output") and isinstance(resp.output, dict):
                    text = resp.output.get("text")

            if text:
                s = str(text).strip()
                return s if s else "未找到匹配的岗位。"
            # 无文本则回退老接口

        # 老接口回退
        response = Generation.call(
            model=model_name,
            messages=messages,
            result_format="text",
            temperature=float(temperature),
            top_p=float(top_p),
            timeout=timeout,
        )
        if getattr(response, "status_code", None) == 200:
            out = response.output
            if isinstance(out, dict) and "text" in out:
                return (out["text"] or "").strip() or "未找到匹配的岗位。"
            if hasattr(response, "output_text"):
                return (response.output_text or "").strip() or "未找到匹配的岗位。"
            return "千问返回为空。"
        else:
            msg = getattr(response, "message", "") or "未知错误"
            meta = f"(model={model_name}, sdk={getattr(dashscope, '__version__', 'unknown')}, base_url={getattr(dashscope, 'base_url', '')})"
            return f"千问调用失败：{msg} {meta}"
    except Exception as e:
        meta = f"(model={model_name}, sdk={getattr(dashscope, '__version__', 'unknown')}, base_url={getattr(dashscope, 'base_url', '')})"
        return f"千问调用异常：{str(e)} {meta}"


def _parse_json_str(s: str) -> Optional[Dict[str, Any]]:
    if not s:
        return None
    try:
        v = json.loads(s)
        if isinstance(v, dict):
            return v
        raise ValueError("必须是 JSON 对象")
    except Exception as e:
        raise ValueError(f"解析 JSON 失败: {e}")


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="岗位检索 + 千问总结（Qwen Plus）")
    parser.add_argument("--q", "--query", dest="query", type=str, default="", help="查询词")
    parser.add_argument("--top_k", type=int, default=0, help="最多返回前 N 条（0 表示不限，使用相似度阈值过滤）")
    parser.add_argument("--min_sim", type=float, default=0.5, help="相似度阈值，返回所有 >= 阈值 的结果（默认 0.5）")
    parser.add_argument("--all", action="store_true", help="返回全部相关结果（忽略 top_k，按相似度阈值过滤）")
    parser.add_argument("--where", type=str, default="", help="元数据过滤 JSON（如 '{\"company\":\"XX\"}'）")
    parser.add_argument("--where_document", type=str, default="", help="文档过滤 JSON")
    parser.add_argument("--no-cache", action="store_true", help="不使用模型/集合缓存")
    parser.add_argument("--json", action="store_true", help="以 JSON 输出检索结果（不调用千问）")
    parser.add_argument("--no-qwen", action="store_true", help="禁用千问总结（仅打印检索结果）")
    parser.add_argument("--model", type=str, default="qwen-plus", help="千问模型名（默认 qwen-plus）")
    parser.add_argument("--temperature", type=float, default=0.2, help="采样温度")
    parser.add_argument("--top_p", type=float, default=0.9, help="核采样 top_p")
    return parser.parse_args(argv)


def run_once_from_args(ns: argparse.Namespace) -> List[Dict[str, Any]]:
    where = _parse_json_str(ns.where) if ns.where else None
    where_doc = _parse_json_str(ns.where_document) if ns.where_document else None
    results = search_jobs(
        query=ns.query,
        top_k=ns.top_k,
        where=where,
        where_document=where_doc,
        include=["documents", "metadatas", "distances"],
        use_cache=not ns.no_cache,
        min_sim=ns.min_sim,
        unlimited=ns.all or ns.top_k <= 0,
    )
    return results


def interactive_loop(
    default_top_k: int = 0,
    use_qwen: bool = True,
    model_name: str = "qwen-plus",
    temperature: float = 0.2,
    top_p: float = 0.9,
    min_sim: float = 0.5,
):
    print("===== 岗位检索+千问回答工具（Qwen Plus）=====")
    print("输入关键词检索岗位（输入 q 退出）。回车直接取全部相关结果（按相似度阈值过滤）。")
    while True:
        try:
            query = input("\n请输入检索词：").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n退出工具！")
            break

        if query.lower() == "q":
            print("退出工具！")
            break
        if not query:
            print("请输入有效关键词！")
            continue

        top_k_in = input("请输入返回条数(回车=全部相关；或输入数字限制数量)：").strip()
        if top_k_in.isdigit():
            top_k = int(top_k_in)
            unlimited = False
        else:
            top_k = 0
            unlimited = True

        try:
            results = search_jobs(
                query=query,
                top_k=top_k,
                min_sim=min_sim,
                unlimited=unlimited,
            )
            if not results:
                print("未检索到相关岗位！")
                continue

            print(f"\n===== 原始检索结果（返回 {len(results)} 条，阈值 >= {min_sim}） =====")
            for res in results:
                sim = res.get("相似度", 0.0)
                text = (res.get("岗位信息", "") or "").strip()
                meta = res.get("元数据", {}) or {}
                show = text[:500] + ("..." if len(text) > 500 else "")
                maybe_id = meta.get("id", "")
                id_part = f"id: {maybe_id} " if maybe_id else ""
                print(f"\n【排名{res['排名']}】{id_part}相似度：{sim}")
                print(f"岗位信息：{show}")
                print(f"元数据：{meta}")

            if use_qwen:
                print(f"\n===== 千问大模型总结（{model_name}） =====")
                qwen_answer = call_qwen_with_search_results(
                    query, results, model_name=model_name, temperature=temperature, top_p=top_p
                )
                print(qwen_answer)
        except Exception as e:
            print(f"检索出错：{e}")


def main():
    ns = parse_args(sys.argv[1:])
    if ns.query:
        try:
            results = run_once_from_args(ns)
        except Exception as e:
            print(f"检索出错：{e}", file=sys.stderr)
            sys.exit(1)

        if ns.json:
            print(json.dumps(results, ensure_ascii=False, indent=2))
            return

        print(f"检索到 {len(results)} 条相关岗位（阈值 >= {ns.min_sim}）：")
        for res in results:
            sim = res.get("相似度", 0.0)
            text = (res.get("岗位信息", "") or "").strip()
            meta = res.get("元数据", {}) or {}
            show = text[:500] + ("..." if len(text) > 500 else "")
            maybe_id = meta.get("id", "")
            id_part = f"id: {maybe_id} " if maybe_id else ""
            print(f"\n【排名{res['排名']}】{id_part}相似度：{sim}")
            print(f"岗位信息：{show}")
            print(f"元数据：{meta}")

        if not ns.no_qwen:
            model_name = _normalize_qwen_model_name(ns.model or "qwen-plus")
            print(f"\n===== 千问大模型总结（{model_name}） =====")
            qwen_answer = call_qwen_with_search_results(
                ns.query, results, model_name=model_name, temperature=ns.temperature, top_p=ns.top_p
            )
            print(qwen_answer)
    else:
        interactive_loop(
            default_top_k=ns.top_k,
            use_qwen=(not ns.no_qwen),
            model_name=_normalize_qwen_model_name(ns.model or "qwen-plus"),
            temperature=ns.temperature,
            top_p=ns.top_p,
            min_sim=ns.min_sim,
        )


if __name__ == "__main__":
    main()
