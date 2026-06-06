# Prompt: Q2D zero-shot (query → pseudo-document)

| | |
|---|---|
| **拡張種別** | クエリ拡張 |
| **適用対象** | Rat2Rev / Rev2Rev test queries (両言語) |
| **出典** | 4F-04 §4.3.4 (Wang+ Query2doc, EMNLP 2023) |
| **canonical source** | [`src/recare_baselines/expansion.py:q2d_zs_messages`](../src/recare_baselines/expansion.py) |
| **モデル** | `gpt-4.1-mini-2025-04-14` (Azure OpenAI) + `Qwen3.5-9B` (dgx03 vLLM, OpenAI-compatible) |
| **パラメータ** | `temperature=0.0`, `max_tokens=512` |
| **連結方式** | BM25: `q×5 + LLM出力` (Wang+ 2023, n=5); dense: `q + " [SEP] " + LLM出力` |
| **出力 path** | [`data/expansion/q2d_zs/{model_slug}_{task}_{lang}_test.jsonl`](../data/expansion/q2d_zs/) |

## Prompt

System: (なし — single user message)

User (EN):
```
Write a passage that answers the following query: {query}
```

User (JA):
```
次のクエリに答える文章を書いてください: {query}
```
