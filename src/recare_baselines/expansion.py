"""Query-side LLM expansion for Issue #10.

Generates Q2D zero-shot, Q2E zero-shot, and task-specific Q2E expansions
for ReCaRe test queries using two LLMs:

- ``gpt-4.1-mini`` via Azure OpenAI (shared with #8 RankGPT)
- ``Qwen3.5-9B`` via dgx03 vLLM (OpenAI-compatible, http://dgx03:8000/v1)

Outputs are saved as JSONL at::

    data/expansion/{prompt_family}/{model_slug}_{task}_{lang}_test.jsonl

with one ``{"qid": ..., "expansion": ...}`` record per test query.

For Rev2Rev's task-specific Q2E (2F-01 §3.3.3), the prompt requires both
``article_before`` and ``article_after``. The qid in Rev2Rev is the
*before*-state article ID; both texts are looked up from
``metadata-{lang}/dataset.jsonl`` keyed on ``article_id_before == qid``
(coverage verified at 100% for both EN and JA test splits, 2026-05-03).
"""

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPANSION_DIR = REPO_ROOT / "data" / "expansion"

# Model registry: slug → (provider, model_id)
# - "azure": uses .env's OPENAI_API_KEY + OPENAI_ENDPOINT, OpenAI Python SDK
# - "vllm": uses dgx03 vLLM, OpenAI-compatible
MODELS = {
    "gpt-4.1-mini": ("azure", "gpt-4.1-mini"),
    "qwen3.5-9b": ("vllm", "Qwen3.5-9B"),
}

VLLM_BASE_URL = "http://dgx03:8000/v1"

PROMPT_FAMILIES = ("q2d_zs", "q2e_zs", "task_q2e")


# ---------------------------------------------------------------------------
# Prompts (see docs/issues/10-augmentation.md §Constraints/プロンプト for source)
# ---------------------------------------------------------------------------


def q2d_zs_messages(query: str, lang: str) -> list[dict]:
    """Q2D zero-shot (4F-04 §4.3.4)."""
    if lang == "en":
        user = f"Write a passage that answers the following query: {query}"
    else:
        user = f"次のクエリに答える文章を書いてください: {query}"
    return [{"role": "user", "content": user}]


def q2e_zs_messages(query: str, lang: str) -> list[dict]:
    """Q2E zero-shot (4F-04 §4.3.3)."""
    if lang == "en":
        user = f"Write a list of keywords for the following query: {query}"
    else:
        user = f"次のクエリに対するキーワードのリストを書いてください: {query}"
    return [{"role": "user", "content": user}]


ARTICLE_MAX_LENGTH = 4000  # 2F-01 §3.3.3 default
MAX_KEYWORDS = 10  # 2F-01 §3.3.3 default


def task_q2e_messages_rev2rev(
    article_before: str,
    article_after: str,
    lang: str,
    max_keywords: int = MAX_KEYWORDS,
    article_max_length: int = ARTICLE_MAX_LENGTH,
) -> list[dict]:
    """Task-specific Q2E for Rev2Rev (verbatim 2F-01 §3.3.3 prompt).

    Source: hourei-search QueryExpansionPrompt (伊藤). Do NOT modify the
    prompt body — it is the canonical legal-expert prompt used in the
    construction paper and we report its effect as-is.
    """
    if not article_after:
        article_after = (
            "(The article has been deleted due to the amendment.)"
            if lang == "en"
            else "（改正により条文を削除）"
        )
    a_before = article_before[:article_max_length]
    a_after = article_after[:article_max_length]
    if lang == "en":
        system = (
            "You are a legal expert. Analyze amendment information and "
            "extract keywords that help find articles likely to be amended together."
        )
        user = f"""
Analyze the following legislative amendment and generate keywords for retrieving articles that are likely to be amended together.

【Article before the amendment】
{a_before}

【Article after the amendment】
{a_after}

Guidelines:
1. Concepts or terms that are typically revised together within the same amendment
2. Terms that belong to legally linked provisions within the same statute
3. Words related to the system or procedure that motivates the revision
4. Terms describing affected stakeholders or regulated subjects

Output up to {max_keywords} keywords, one per line.
"""
    else:
        system = (
            "あなたは法律の専門家です。"
            "改正情報から、同時に改正される可能性のある条文を検索するための"
            "キーワードを抽出してください。"
        )
        user = f"""
以下の法律改正情報を分析して、同時に改正される可能性のある条文を検索するためのキーワードを生成してください。

【改正前の条文】
{a_before}

【改正後の条文】
{a_after}

キーワード生成の指針：
1. 同じ法改正で変更される可能性の高い関連概念や用語
2. 法律の構造上、連動して変更される条文に含まれる可能性の高い単語
3. 改正の背景となる制度や手続きに関連する用語
4. 改正によって影響を受ける関係者や対象に関する用語

キーワードは1行に1つずつ、最大{max_keywords}個まで出力してください。
"""
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def task_q2e_messages_rat2rev(
    rationale: str,
    lang: str,
    max_keywords: int = MAX_KEYWORDS,
    article_max_length: int = ARTICLE_MAX_LENGTH,
) -> list[dict]:
    """Task-specific Q2E for Rat2Rev (new prompt, structurally derived from
    2F-01 §3.3.3 with rationale instead of pre/post article pair).
    """
    rationale = rationale[:article_max_length]
    if lang == "en":
        system = (
            "You are a legal expert. Given the rationale of a proposed amendment, "
            "extract keywords that help retrieve articles likely to be amended "
            "to implement that rationale."
        )
        user = f"""
Analyze the following amendment rationale and generate keywords for retrieving articles that would be amended for it.

【Amendment rationale】
{rationale}

Guidelines:
1. Systems, concepts, and terminology that the rationale references
2. Words likely to appear in articles co-amended in the same legislative package
3. Social issues, stakeholders, and procedures that motivate the amendment
4. Names of affected statutes or chapters explicitly mentioned

Output up to {max_keywords} keywords, one per line.
"""
    else:
        system = (
            "あなたは法律の専門家です。"
            "改正理由文から、この改正で同時に変更される可能性のある条文を"
            "検索するためのキーワードを抽出してください。"
        )
        user = f"""
以下の改正理由文を分析して、この改正で同時に変更される可能性のある条文を検索するためのキーワードを生成してください。

【改正理由文】
{rationale}

キーワード生成の指針：
1. 改正理由が言及する制度・概念・用語
2. 同じ改正で連動して修正される可能性の高い条文に含まれる語
3. 改正の背景となる社会的課題・ステークホルダー
4. 影響を受ける法律名・章節 (改正理由文中で明示されている場合)

キーワードは1行に1つずつ、最大{max_keywords}個まで出力してください。
"""
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def build_messages(
    prompt_family: str,
    *,
    query: str,
    task: str,
    lang: str,
    article_before: str | None = None,
    article_after: str | None = None,
) -> list[dict]:
    """Build chat messages for one (prompt_family, task, lang) cell.

    For ``task_q2e`` on Rev2Rev, ``article_before`` and ``article_after``
    must be provided (looked up from metadata-{lang} by article_id_before == qid).
    For ``task_q2e`` on Rat2Rev, only ``query`` (the rationale) is needed.
    """
    if prompt_family == "q2d_zs":
        return q2d_zs_messages(query, lang)
    if prompt_family == "q2e_zs":
        return q2e_zs_messages(query, lang)
    if prompt_family == "task_q2e":
        if task == "rev2rev":
            # Both kwargs MUST be passed by the caller — load_rev2rev_article_pairs
            # coerces JSON-null text_before/after to empty string so the
            # 2F-01 deletion-fallback in task_q2e_messages_rev2rev triggers.
            if article_before is None or article_after is None:
                raise ValueError(
                    "task_q2e on rev2rev requires article_before and article_after "
                    "kwargs (use load_rev2rev_article_pairs())"
                )
            return task_q2e_messages_rev2rev(article_before, article_after, lang)
        if task == "rat2rev":
            return task_q2e_messages_rat2rev(query, lang)
        raise ValueError(f"unknown task: {task}")
    raise ValueError(f"unknown prompt_family: {prompt_family}")


# ---------------------------------------------------------------------------
# Metadata lookup (article_before / article_after for Rev2Rev)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Retrieval-time query concatenation (Phase 2)
#
# Wang+ Query2doc (EMNLP 2023, arxiv:2303.07678) and Jagerman+ 2023 (arxiv:
# 2305.03653) both apply n=5 query repetition for BM25; Wang+ uses [SEP]
# separator for dense. We follow that protocol verbatim.
# ---------------------------------------------------------------------------


N_REPEAT_BM25 = 5
DENSE_SEP_TOKEN = " [SEP] "


def load_query_expansions(path: Path) -> dict[str, str]:
    """Load ``{qid: expansion}`` from a JSONL produced by :func:`run_cell`.

    Records with empty ``expansion`` (e.g. content_filter rejections) are
    returned with an empty string so the caller can decide how to handle
    them (typically: fall back to the unexpanded query).
    """
    out: dict[str, str] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            qid = rec.get("qid")
            if qid is None:
                continue
            out[qid] = rec.get("expansion", "") or ""
    return out


def apply_query_expansion(
    queries: dict[str, str],
    expansion_path: Path,
    *,
    mode: str,
) -> tuple[dict[str, str], dict[str, int]]:
    """Apply Wang/Jagerman concatenation to every query in ``queries``.

    Parameters
    ----------
    queries
        ``{qid: original query text}``.
    expansion_path
        JSONL of ``{"qid": ..., "expansion": ...}`` records.
    mode
        ``"bm25_n5"`` → ``q⁺ = " ".join([q]*5 + [expansion])`` per Wang/Jagerman.
        ``"dense_sep"`` → ``q⁺ = q + " [SEP] " + expansion`` per Wang Query2doc.

    Returns
    -------
    expanded_queries, stats
        ``stats`` has counts ``{"applied": int, "missing": int, "empty": int}``
        — empty/missing qids fall back to the unexpanded query.
    """
    if mode not in {"bm25_n5", "dense_sep"}:
        raise ValueError(f"unknown mode: {mode!r} (expected 'bm25_n5' or 'dense_sep')")
    expansions = load_query_expansions(Path(expansion_path))
    stats = {"applied": 0, "missing": 0, "empty": 0}
    out: dict[str, str] = {}
    for qid, q in queries.items():
        if qid not in expansions:
            stats["missing"] += 1
            out[qid] = q
            continue
        ext = expansions[qid]
        if not ext:
            stats["empty"] += 1
            out[qid] = q  # fallback: unexpanded (e.g. content_filter rejection)
            continue
        if mode == "bm25_n5":
            out[qid] = " ".join([q] * N_REPEAT_BM25 + [ext])
        else:  # dense_sep
            out[qid] = f"{q}{DENSE_SEP_TOKEN}{ext}"
        stats["applied"] += 1
    return out, stats


def derive_persist_suffix(expansion_path: Path) -> str:
    """Derive a ``+suffix`` for model_key persistence from the JSONL path.

    Path layout: ``data/expansion/<prompt_family>/<model_slug>_<task>_<lang>_test.jsonl``.
    Suffix returned: ``<prompt_family>.<model_slug>`` (e.g. ``q2d_zs.gpt-4.1-mini``).
    """
    p = Path(expansion_path)
    prompt_family = p.parent.name
    fname = p.stem  # e.g. "gpt-4.1-mini_rat2rev_en_test"
    # split off the trailing "_<task>_<lang>_<split>" — task/lang/split are
    # known so we can reverse-pop them.
    parts = fname.split("_")
    if len(parts) >= 4 and parts[-1] in {"train", "validation", "test"}:
        # parts[-1]=split, parts[-2]=lang, parts[-3]=task, model = rest
        model_slug = "_".join(parts[:-3])
    else:
        model_slug = fname
    return f"{prompt_family}.{model_slug}"


def load_rev2rev_article_pairs(lang: str) -> dict[str, tuple[str, str]]:
    """Return ``{qid: (text_before, text_after)}`` for Rev2Rev.

    The Rev2Rev qid is the *before*-state article ID, so we key the
    metadata lookup on ``article_id_before``. Both EN and JA cover 100%
    of the test split (verified 2026-05-03).

    ``text_after`` is JSON-null in the metadata when the amendment deleted
    the article (~5% of test qids). We coerce that to an empty string here
    so :func:`task_q2e_messages_rev2rev` can apply its 2F-01 deletion-fallback
    ("(The article has been deleted due to the amendment.)" /
    "（改正により条文を削除）") downstream rather than hitting the
    ``None``-guard in :func:`build_messages`.
    """
    from .data import _hf_path

    path = _hf_path(f"metadata-{lang}/dataset.jsonl")
    out: dict[str, tuple[str, str]] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            bid = rec.get("article_id_before")
            if not bid:
                continue
            out[bid] = (
                rec.get("text_before") or "",
                rec.get("text_after") or "",
            )
    return out


# ---------------------------------------------------------------------------
# Async client
# ---------------------------------------------------------------------------


@dataclass
class ClientCfg:
    provider: str  # "azure" | "vllm"
    model: str
    base_url: str | None = None
    api_key: str | None = None
    api_version: str | None = None  # azure only


def _resolve_cfg(model_slug: str) -> ClientCfg:
    if model_slug not in MODELS:
        raise ValueError(f"unknown model_slug: {model_slug} (have {list(MODELS)})")
    provider, model_id = MODELS[model_slug]
    if provider == "azure":
        try:
            from dotenv import load_dotenv

            load_dotenv(REPO_ROOT / ".env", override=False)
        except ImportError:
            pass
        api_key = os.environ.get("OPENAI_API_KEY")
        endpoint = os.environ.get("OPENAI_ENDPOINT") or os.environ.get(
            "AZURE_OPENAI_ENDPOINT"
        )
        if not api_key or not endpoint:
            raise RuntimeError(
                "Azure config missing: set OPENAI_API_KEY + OPENAI_ENDPOINT in .env"
            )
        return ClientCfg(
            provider="azure",
            model=model_id,
            base_url=endpoint,
            api_key=api_key,
            api_version=os.environ.get("OPENAI_API_VERSION", "2024-10-21"),
        )
    if provider == "vllm":
        return ClientCfg(
            provider="vllm",
            model=model_id,
            base_url=VLLM_BASE_URL,
            api_key="EMPTY",
        )
    raise ValueError(f"unknown provider: {provider}")


def _build_async_client(cfg: ClientCfg):
    if cfg.provider == "azure":
        from openai import AsyncAzureOpenAI

        return AsyncAzureOpenAI(
            api_key=cfg.api_key,
            azure_endpoint=cfg.base_url,
            api_version=cfg.api_version,
        )
    from openai import AsyncOpenAI

    return AsyncOpenAI(base_url=cfg.base_url, api_key=cfg.api_key)


async def _generate_one(
    client,
    cfg: ClientCfg,
    qid: str,
    messages: list[dict],
    *,
    max_tokens: int,
    temperature: float,
    semaphore: asyncio.Semaphore,
    max_retries: int = 5,
) -> tuple[str, str, str | None]:
    """Return ``(qid, expansion, error_marker)``.

    ``error_marker`` is ``None`` on success, otherwise a short tag such as
    ``"content_filter"`` or ``"max_retries_exceeded"``. Non-retriable Azure
    content-filter rejections produce ``("", "content_filter")`` so we
    keep going for the remaining 99% of queries — losing 1-2 queries to
    Azure RAI on legal text is preferable to aborting the whole cell.
    """
    from openai import APIError, BadRequestError

    async with semaphore:
        for attempt in range(max_retries):
            try:
                resp = await client.chat.completions.create(
                    model=cfg.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                return qid, resp.choices[0].message.content or "", None
            except BadRequestError as e:
                code = getattr(e, "code", None) or (
                    (e.body or {}).get("error", {}).get("code") if hasattr(e, "body") else None
                )
                if code == "content_filter":
                    return qid, "", "content_filter"
                # Non-content-filter 400 is a code bug; surface it
                raise
            except APIError:
                # Transient API errors (429, 500, 503, network) — retry
                if attempt + 1 >= max_retries:
                    return qid, "", "max_retries_exceeded"
                await asyncio.sleep(min(60.0, 2.0 ** attempt))
            except Exception:
                if attempt + 1 >= max_retries:
                    return qid, "", "unknown_error"
                await asyncio.sleep(min(60.0, 2.0 ** attempt))
        return qid, "", "max_retries_exceeded"


# ---------------------------------------------------------------------------
# Top-level run
# ---------------------------------------------------------------------------


def _output_path(prompt_family: str, model_slug: str, task: str, lang: str) -> Path:
    return (
        EXPANSION_DIR
        / prompt_family
        / f"{model_slug}_{task}_{lang}_test.jsonl"
    )


def _existing_qids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    out: set[str] = set()
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if rec.get("expansion"):
                    out.add(rec["qid"])
    except json.JSONDecodeError:
        return set()  # treat partial as nothing — caller will overwrite
    return out


async def _run_async(
    *,
    prompt_family: str,
    model_slug: str,
    task: str,
    lang: str,
    queries: dict[str, str],
    article_pairs: dict[str, tuple[str, str]] | None,
    max_tokens: int,
    temperature: float,
    concurrency: int,
    skip_existing: bool,
):
    cfg = _resolve_cfg(model_slug)
    out_path = _output_path(prompt_family, model_slug, task, lang)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = _existing_qids(out_path) if skip_existing else set()
    pending = [(qid, q) for qid, q in queries.items() if qid not in done]
    if not pending:
        print(f"  nothing to do (all {len(queries)} qids already in {out_path.name})")
        return

    print(
        f"  {prompt_family} {model_slug} {task}/{lang}: {len(pending)} queries "
        f"(skipping {len(done)} done)"
    )
    client = _build_async_client(cfg)
    semaphore = asyncio.Semaphore(concurrency)
    try:
        from tqdm.asyncio import tqdm_asyncio

        coros = []
        for qid, q in pending:
            kwargs = {"query": q, "task": task, "lang": lang}
            if prompt_family == "task_q2e" and task == "rev2rev":
                pair = (article_pairs or {}).get(qid)
                if pair is None:
                    raise RuntimeError(
                        f"qid {qid} not in metadata-{lang}; cannot build "
                        f"rev2rev task_q2e prompt"
                    )
                kwargs["article_before"], kwargs["article_after"] = pair
            messages = build_messages(prompt_family, **kwargs)
            coros.append(
                _generate_one(
                    client,
                    cfg,
                    qid,
                    messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    semaphore=semaphore,
                )
            )
        results = await tqdm_asyncio.gather(
            *coros, desc=f"{prompt_family}:{model_slug}:{task}/{lang}"
        )
    finally:
        await client.close()

    # Read pre-existing JSONL (may carry success records and prior errors).
    existing: dict[str, dict] = {}
    if out_path.exists():
        with open(out_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "qid" in rec:
                    existing[rec["qid"]] = rec

    # Merge new results. Prefer success over error; preserve error markers.
    n_ok = n_err = 0
    err_breakdown: dict[str, int] = {}
    for qid, expansion, err in results:
        rec = {"qid": qid, "expansion": expansion}
        if err is not None:
            rec["error"] = err
            n_err += 1
            err_breakdown[err] = err_breakdown.get(err, 0) + 1
        else:
            n_ok += 1
        existing[qid] = rec

    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        for qid in sorted(existing):
            f.write(json.dumps(existing[qid], ensure_ascii=False) + "\n")
    tmp_path.replace(out_path)
    print(
        f"  wrote {out_path} ({len(existing)} records: ok={n_ok}, err={n_err} "
        f"breakdown={err_breakdown})"
    )


def run_cell(
    *,
    prompt_family: str,
    model_slug: str,
    task: str,
    lang: str,
    queries: dict[str, str],
    article_pairs: dict[str, tuple[str, str]] | None = None,
    max_tokens: int = 512,
    temperature: float = 0.0,
    concurrency: int = 16,
    skip_existing: bool = True,
):
    """Generate expansions for one (prompt_family, model, task, lang) cell.

    For ``prompt_family='task_q2e'`` and ``task='rev2rev'``, pass
    ``article_pairs`` from :func:`load_rev2rev_article_pairs`.
    """
    if prompt_family not in PROMPT_FAMILIES:
        raise ValueError(f"unknown prompt_family: {prompt_family}")
    asyncio.run(
        _run_async(
            prompt_family=prompt_family,
            model_slug=model_slug,
            task=task,
            lang=lang,
            queries=queries,
            article_pairs=article_pairs,
            max_tokens=max_tokens,
            temperature=temperature,
            concurrency=concurrency,
            skip_existing=skip_existing,
        )
    )
