# Prompt: Q2E zero-shot (query → keyword list)

| | |
|---|---|
| **拡張種別** | クエリ拡張 |
| **適用対象** | Rat2Rev / Rev2Rev test queries (両言語) |
| **出典** | 4F-04 §4.3.3 (Jagerman+ 2023, *Query Expansion by Prompting LLMs*) |
| **canonical source** | [`src/recare_baselines/expansion.py:q2e_zs_messages`](../src/recare_baselines/expansion.py) |
| **モデル** | `gpt-4.1-mini-2025-04-14` (Azure OpenAI) + `Qwen3.5-9B` (dgx03 vLLM, OpenAI-compatible) |
| **パラメータ** | `temperature=0.0`, `max_tokens=512` |
| **連結方式** | BM25: `q×5 + LLM出力` (Jagerman+ 2023 / Wang+ 2023, n=5); dense: `q + " [SEP] " + LLM出力` (Q2E は本来 BM25 専用だが ablation として dense にも適用) |
| **出力 path** | [`data/expansion/q2e_zs/{model_slug}_{task}_{lang}_test.jsonl`](../data/expansion/q2e_zs/) |

## Prompt

System: (なし — single user message)

User (EN):
```
Write a list of keywords for the following query: {query}
```

User (JA):
```
次のクエリに対するキーワードのリストを書いてください: {query}
```
