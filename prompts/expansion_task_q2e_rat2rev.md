# Prompt: タスク特化 Q2E — Rat2Rev (rationale-driven keyword extraction)

| | |
|---|---|
| **拡張種別** | クエリ拡張 (タスク特化) |
| **適用対象** | Rat2Rev test queries (両言語) |
| **出典** | 新規 (本研究, 2F-01 §3.3.3 の構造を rationale 単独入力用に派生) |
| **canonical source** | [`src/recare_baselines/expansion.py:task_q2e_messages_rat2rev`](../src/recare_baselines/expansion.py) |
| **モデル** | `gpt-4.1-mini-2025-04-14` (Azure OpenAI) + `Qwen3.5-9B` (dgx03 vLLM) |
| **パラメータ** | `temperature=0.0`, `max_tokens=512`, `max_keywords=10`, `article_max_length=4000` |
| **入力** | `rationale` (= test query text. Rev2Rev と異なり pre/post の article ペアは存在しない) |
| **連結方式** | BM25: `q×5 + LLM出力`; dense: `q + " [SEP] " + LLM出力` |
| **出力 path** | [`data/expansion/task_q2e/{model_slug}_rat2rev_{lang}_test.jsonl`](../data/expansion/task_q2e/) |

## Prompt

### EN

System:
```
You are a legal expert. Given the rationale of a proposed amendment, extract keywords that help retrieve articles likely to be amended to implement that rationale.
```

User:
```
Analyze the following amendment rationale and generate keywords for retrieving articles that would be amended for it.

【Amendment rationale】
{rationale}

Guidelines:
1. Systems, concepts, and terminology that the rationale references
2. Words likely to appear in articles co-amended in the same legislative package
3. Social issues, stakeholders, and procedures that motivate the amendment
4. Names of affected statutes or chapters explicitly mentioned

Output up to {max_keywords} keywords, one per line.
```

### JA

System:
```
あなたは法律の専門家です。改正理由文から、この改正で同時に変更される可能性のある条文を検索するためのキーワードを抽出してください。
```

User:
```
以下の改正理由文を分析して、この改正で同時に変更される可能性のある条文を検索するためのキーワードを生成してください。

【改正理由文】
{rationale}

キーワード生成の指針：
1. 改正理由が言及する制度・概念・用語
2. 同じ改正で連動して修正される可能性の高い条文に含まれる語
3. 改正の背景となる社会的課題・ステークホルダー
4. 影響を受ける法律名・章節 (改正理由文中で明示されている場合)

キーワードは1行に1つずつ、最大{max_keywords}個まで出力してください。
```

## 改訂履歴

- 2026-05-04: JA Guideline 2 を「同じ改正パッケージで連動して」→「同じ改正で連動して」(1 字 trim, 表現の冗長を除去). 該当 cell (`task_q2e × {gpt, qwen} × rat2rev × ja`) の生成 + BM25/jina-v3 評価を再実行済 (commit `5cbdf5a`).
