# Prompt: LLM 文書拡張 (legal-expert document explanation)

| | |
|---|---|
| **拡張種別** | 文書拡張 |
| **適用対象** | コーパス全件 (`corpus-en` 91,361 docs + `corpus-ja` 90,170 docs) |
| **出典** | 2F-01 §3.3.4 (伊藤+ DEIM 2026) — 「文書拡張プロンプト最適化」の **初期プロンプト** (最適化前の baseline). 本研究はこの初期プロンプトをそのまま使用 (本研究のスコープは GEPA 最適化版ではなく初期版での効果測定) |
| **canonical source** | hourei-search project の `DocumentExpansionPrompt` クラス (伊藤+ 内製). 本 repo では生成済 JSONL を再利用 (再実行不要) |
| **モデル** | `openai/gpt-4.1-mini-2025-04-14` (Azure / OpenAI) |
| **パラメータ** | `temperature=0.0`, `article_max_length=4000` |
| **生成済データ path** | [`data/recare_d2e/recare_en_d2e_documents.jsonl`](../data/recare_d2e/) (91,361 records, ID 100% 整合) <br> [`data/recare_d2e/recare_ja_d2e_documents.jsonl`](../data/recare_d2e/) (90,170 records, ID 100% 整合) |
| **JSONL スキーマ** | `{"id": "<docid>", "contents": "<augmented text>"}` (= 拡張文のみ. 元 doc text は含まない) |
| **連結方式** | 索引/エンコード時に `元 doc + " " + 拡張 text` で結合してから BM25 Lucene index / jina-v3 embedding を構築 |

## Prompt

### EN

System:
```
You are a legal expert. Extract the core content of the given provision and the social purpose behind it, and generate an explanatory text that makes it easier to analyze its relationship with other provisions.
```

User:
```
Analyze the following article and explain it according to the specified guidelines.

【Article】
{document_text}

Guidelines for Explanation Generation:
1. Explain the core content of what this provision establishes.
2. Explain the legislative intent and social purpose the provision aims to achieve, considering what social issues or demands it addresses.

Based on the original text and utilizing legal expertise, generate the explanation in a single paragraph.
```

### JA

System:
```
あなたは法律の専門家です。与えられた条文の核心的な内容と、その背景にある社会的な目的を抽出し、他の条文との関連性を分析しやすくするための説明文を生成してください。
```

User:
```
以下の条文を分析し、指定された項目に従って解説してください。

【条文】
{document_text}

説明文生成の指針：
1. この条文が何を定めているのか、その核心的な内容を説明してください。
2. この条文がどのような社会的な課題や要請に応えるために存在すると考えられるか、 その立法趣旨や条文が目指している社会的な目的を説明してください。

元の文書内容を基に、法律の専門知識を活用して、説明文を1つの段落で生成してください。
```

`{document_text}` は `[: article_max_length]` でヘッド切り捨て (default 4000 chars).

## 利用箇所

#10 augmentation の `+d2e` モード (BM25, jina-v3 双方). 主要 finding (`results/augmentation.md`): **本研究で唯一 BM25 を大幅改善した拡張** (Rat2Rev EN R@100 +0.066, R@1000 +0.112, MAP +0.020).
