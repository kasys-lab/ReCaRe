"""Cross-encoder rerankers for ReCaRe (Issue #7).

Reranks the top-K candidates from a first-stage retriever by scoring every
``(query, candidate-doc)`` pair with a cross-encoder. We use BM25 top-100 as
the sole first stage per project decision (issue #7 brief).

Models (multilingual cross-encoders):

* ``bge-reranker-v2-m3``  — ``BAAI/bge-reranker-v2-m3``: XLM-RoBERTa head,
  ``AutoModelForSequenceClassification`` with one logit per pair (sigmoid).
* ``jina-reranker-v2``    — ``jinaai/jina-reranker-v2-base-multilingual``:
  custom remote-code XLM-RoBERTa flash, ``AutoModelForSequenceClassification``
  with one logit per pair.
* ``qwen3-reranker-{0.6b,4b,8b}`` — ``Qwen/Qwen3-Reranker-{0.6B,4B,8B}``: causal
  LM with chat-template prompt; the score is the *logit difference*
  ``logit("yes") − logit("no")`` at the final-position assistant token.

All implementations use raw ``transformers`` (no sentence-transformers) for
maximum compatibility with our pinned ``transformers==4.57``. fp16 on CUDA.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
from tqdm import tqdm

from . import data

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RerankerSpec:
    key: str
    hf_id: str
    family: Literal["seqcls", "qwen3-lm"]
    max_length: int = 8192
    trust_remote_code: bool = False
    batch_size: int = 16


MODELS: dict[str, RerankerSpec] = {
    "bge-reranker-v2-m3": RerankerSpec(
        key="bge-reranker-v2-m3",
        hf_id="BAAI/bge-reranker-v2-m3",
        family="seqcls",
        max_length=8192,
        batch_size=16,
    ),
    "jina-reranker-v2": RerankerSpec(
        key="jina-reranker-v2",
        hf_id="jinaai/jina-reranker-v2-base-multilingual",
        family="seqcls",
        max_length=1024,  # Jina Reranker v2's published context cap
        trust_remote_code=True,
        batch_size=16,
    ),
    "qwen3-reranker-0.6b": RerankerSpec(
        key="qwen3-reranker-0.6b",
        hf_id="Qwen/Qwen3-Reranker-0.6B",
        family="qwen3-lm",
        max_length=8192,
        batch_size=16,
    ),
    "qwen3-reranker-4b": RerankerSpec(
        key="qwen3-reranker-4b",
        hf_id="Qwen/Qwen3-Reranker-4B",
        family="qwen3-lm",
        max_length=8192,
        batch_size=4,
    ),
    "qwen3-reranker-8b": RerankerSpec(
        key="qwen3-reranker-8b",
        hf_id="Qwen/Qwen3-Reranker-8B",
        family="qwen3-lm",
        max_length=8192,
        batch_size=2,
    ),
}


_QWEN3_INSTRUCTION = (
    "Given a web search query, retrieve relevant passages that answer the query"
)
_QWEN3_PREFIX = (
    "<|im_start|>system\nJudge whether the Document meets the requirements "
    'based on the Query and the Instruct provided. Note that the answer can '
    'only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
)
_QWEN3_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def _device():
    return "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Family: seqcls (BGE-Reranker-v2-M3, Jina Reranker v2)
# ---------------------------------------------------------------------------


class _SeqClsReranker:
    """``AutoModelForSequenceClassification`` cross-encoder with one logit/pair."""

    def __init__(self, spec: RerankerSpec):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.spec = spec
        self.tokenizer = AutoTokenizer.from_pretrained(
            spec.hf_id, trust_remote_code=spec.trust_remote_code
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            spec.hf_id, trust_remote_code=spec.trust_remote_code
        )
        self.model.eval().to(_device())
        if _device() == "cuda":
            self.model = self.model.half()

    @torch.no_grad()
    def score_pairs(self, pairs: list[tuple[str, str]]) -> list[float]:
        if not pairs:
            return []
        texts_a = [q for q, _ in pairs]
        texts_b = [d for _, d in pairs]
        inputs = self.tokenizer(
            texts_a,
            texts_b,
            padding=True,
            truncation=True,
            max_length=self.spec.max_length,
            return_tensors="pt",
        )
        inputs = {k: v.to(_device()) for k, v in inputs.items()}
        out = self.model(**inputs, return_dict=True)
        logits = out.logits.view(-1).float().cpu().numpy()
        return logits.tolist()


# ---------------------------------------------------------------------------
# Family: qwen3-lm (Qwen3-Reranker-{0.6B, 4B, 8B})
# ---------------------------------------------------------------------------


class _Qwen3Reranker:
    """Qwen3-Reranker: chat-template prompt, score = logit('yes') − logit('no')."""

    def __init__(self, spec: RerankerSpec):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.spec = spec
        self.tokenizer = AutoTokenizer.from_pretrained(spec.hf_id, padding_side="left")
        torch_dtype = torch.float16 if _device() == "cuda" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            spec.hf_id, torch_dtype=torch_dtype
        )
        self.model.eval().to(_device())

        # The "yes" / "no" tokens appear with a leading space in Qwen3's tokenizer.
        self.token_yes = self.tokenizer(" yes", add_special_tokens=False).input_ids[-1]
        self.token_no = self.tokenizer(" no", add_special_tokens=False).input_ids[-1]
        # Pre-tokenize the static prefix/suffix for efficiency.
        self.prefix_ids = self.tokenizer.encode(_QWEN3_PREFIX, add_special_tokens=False)
        self.suffix_ids = self.tokenizer.encode(_QWEN3_SUFFIX, add_special_tokens=False)

    def _format_pair(self, query: str, doc: str) -> str:
        return f"<Instruct>: {_QWEN3_INSTRUCTION}\n<Query>: {query}\n<Document>: {doc}"

    @torch.no_grad()
    def score_pairs(self, pairs: list[tuple[str, str]]) -> list[float]:
        if not pairs:
            return []
        # Encode the variable middle, leaving room for prefix + suffix.
        budget = self.spec.max_length - len(self.prefix_ids) - len(self.suffix_ids)
        body_texts = [self._format_pair(q, d) for q, d in pairs]
        body_enc = self.tokenizer(
            body_texts,
            add_special_tokens=False,
            truncation=True,
            max_length=max(budget, 1),
            padding=False,
        )["input_ids"]
        # Concatenate prefix + body + suffix per example, then pad on the left.
        ids_per_ex = [self.prefix_ids + b + self.suffix_ids for b in body_enc]
        max_len = max(len(x) for x in ids_per_ex)
        pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        input_ids = torch.full(
            (len(ids_per_ex), max_len), pad_id, dtype=torch.long, device=_device()
        )
        attention_mask = torch.zeros(
            (len(ids_per_ex), max_len), dtype=torch.long, device=_device()
        )
        for i, ids in enumerate(ids_per_ex):
            input_ids[i, max_len - len(ids) :] = torch.tensor(ids, device=_device())
            attention_mask[i, max_len - len(ids) :] = 1
        out = self.model(input_ids=input_ids, attention_mask=attention_mask)
        # Logits at the last token position give next-token distribution.
        last_logits = out.logits[:, -1, :]
        score = (last_logits[:, self.token_yes] - last_logits[:, self.token_no]).float()
        return score.cpu().numpy().tolist()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _load(spec: RerankerSpec):
    if spec.family == "seqcls":
        return _SeqClsReranker(spec)
    if spec.family == "qwen3-lm":
        return _Qwen3Reranker(spec)
    raise ValueError(f"unknown family {spec.family!r}")


def load_corpus_texts(lang: str) -> dict[str, str]:
    """Materialise the language-specific corpus into ``{doc_id: text}``."""
    path = data._hf_path(f"corpus-{lang}/corpus.jsonl")
    out: dict[str, str] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            out[rec["_id"]] = rec.get("text", "")
    return out


def load_first_stage(
    first_stage: str, task: str, lang: str
) -> dict[str, list[tuple[str, float]]]:
    """Load a first-stage top-K run-file."""
    repo_root = Path(__file__).resolve().parents[2]
    if first_stage == "bm25":
        path = (
            repo_root
            / "results"
            / "intermediate"
            / "bm25_top100"
            / f"{task}_{lang}.jsonl"
        )
    else:
        path = (
            repo_root
            / "results"
            / "intermediate"
            / "dense_top100"
            / f"{task}_{lang}_{first_stage}.jsonl"
        )
    if not path.exists():
        raise FileNotFoundError(f"first-stage run-file missing: {path}")
    out: dict[str, list[tuple[str, float]]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            out[rec["qid"]] = [(d, float(s)) for d, s in rec["results"]]
    return out


def rerank_run(
    model_key: str,
    queries: dict[str, str],
    candidates: dict[str, list[tuple[str, float]]],
    corpus_texts: dict[str, str],
) -> dict[str, list[tuple[str, float]]]:
    """Score every (query, candidate-doc) pair and re-sort."""
    spec = MODELS[model_key]
    reranker = _load(spec)

    out: dict[str, list[tuple[str, float]]] = {}
    for qid in tqdm(queries.keys(), desc=f"rerank:{model_key}", unit="q"):
        cand = candidates.get(qid)
        if not cand:
            out[qid] = []
            continue
        q_text = queries[qid]
        d_ids: list[str] = []
        pairs: list[tuple[str, str]] = []
        for did, _s in cand:
            d_text = corpus_texts.get(did, "")
            if not d_text:
                continue
            d_ids.append(did)
            pairs.append((q_text, d_text))
        if not pairs:
            out[qid] = []
            continue
        scores: list[float] = []
        for start in range(0, len(pairs), spec.batch_size):
            batch = pairs[start : start + spec.batch_size]
            scores.extend(reranker.score_pairs(batch))
        scored = sorted(zip(d_ids, scores), key=lambda x: x[1], reverse=True)
        out[qid] = scored
    return out
