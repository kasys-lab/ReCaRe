# Prompt: タスク特化 Q2E — Rev2Rev (legal-expert keyword extraction)

| | |
|---|---|
| **拡張種別** | クエリ拡張 (タスク特化) |
| **適用対象** | Rev2Rev test queries (両言語) |
| **出典** | 2F-01 §3.3.3 (伊藤+ DEIM 2026 *文書拡張プロンプト最適化に基づく同時更新文書検索*) — オリジナルを変更せず再現 |
| **canonical source** | [`src/recare_baselines/expansion.py:task_q2e_messages_rev2rev`](../src/recare_baselines/expansion.py) |
| **モデル** | `gpt-4.1-mini-2025-04-14` (Azure OpenAI) + `Qwen3.5-9B` (dgx03 vLLM) |
| **パラメータ** | `temperature=0.0`, `max_tokens=512`, `max_keywords=10`, `article_max_length=4000` |
| **入力** | `(article_before, article_after)` を [`metadata-{lang}/dataset.jsonl`](https://huggingface.co/datasets/kasys/ReCaRe) から `article_id_before == qid` で lookup. `text_after` が JSON-null の場合は削除フォールバック文字列を埋める (オリジナル踏襲) |
| **連結方式** | BM25: `q×5 + LLM出力`; dense: `q + " [SEP] " + LLM出力` |
| **出力 path** | [`data/expansion/task_q2e/{model_slug}_rev2rev_{lang}_test.jsonl`](../data/expansion/task_q2e/) |

## Prompt

### EN

System:
```
You are a legal expert. Analyze amendment information and extract keywords that help find articles likely to be amended together.
```

User:
```
Analyze the following legislative amendment and generate keywords for retrieving articles that are likely to be amended together.

【Article before the amendment】
{article_before}

【Article after the amendment】
{article_after}

Guidelines:
1. Concepts or terms that are typically revised together within the same amendment
2. Terms that belong to legally linked provisions within the same statute
3. Words related to the system or procedure that motivates the revision
4. Terms describing affected stakeholders or regulated subjects

Output up to {max_keywords} keywords, one per line.
```

削除フォールバック (`article_after` が空の場合): `(The article has been deleted due to the amendment.)`

### JA

System:
```
あなたは法律の専門家です。改正情報から、同時に改正される可能性のある条文を検索するためのキーワードを抽出してください。
```

User:
```
以下の法律改正情報を分析して、同時に改正される可能性のある条文を検索するためのキーワードを生成してください。

【改正前の条文】
{article_before}

【改正後の条文】
{article_after}

キーワード生成の指針：
1. 同じ法改正で変更される可能性の高い関連概念や用語
2. 法律の構造上、連動して変更される条文に含まれる可能性の高い単語
3. 改正の背景となる制度や手続きに関連する用語
4. 改正によって影響を受ける関係者や対象に関する用語

キーワードは1行に1つずつ、最大{max_keywords}個まで出力してください。
```

削除フォールバック (`article_after` が空の場合): `（改正により条文を削除）`
