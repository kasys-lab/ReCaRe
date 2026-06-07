# ReCaRe Dataset Construction

This document describes the procedure used to construct ReCaRe — its
corpus, queries, qrels, splits, and validation sample — from public
legislative sources.

## 1. Task Formulation

ReCaRe contains an article-level corpus `D` and, for each query `q` in
each task, a set of relevant articles `D⁺_q ⊂ D`.

The source data is organized around **amendment events**. An amendment
event `e` comprises:

1. an **amending act** — the legal instrument that effects the change,
2. an **amendment rationale** `r_e` — text stating the motivation or
   background of the amendment,
3. the **amended acts** — the legal instruments modified by the amending
   act,
4. the **target articles** `A_e ⊂ D` — the pre-revision versions of the
   individual articles modified by the amending act.

Two retrieval tasks are defined over the same set of events.

**Rat2Rev.** Given the amendment rationale `r_e` as the query `q`,
retrieve the target articles from `D`, where `D⁺_q = A_e`. Each
amendment event yields exactly one Rat2Rev query.

**Rev2Rev.** Given an observed revision of a target article `a ∈ A_e`
as the query `q`, represented by its pre- and post-revision versions,
retrieve the other target articles in the same event from `D \ {a}`,
where `D⁺_q = A_e \ {a}`. Only events with `|A_e| ≥ 2` yield Rev2Rev
queries; from each such event, up to **five** queries are constructed by
sampling `min(|A_e|, 5)` target articles uniformly at random.

## 2. Sources

| | EU | JA |
|---|---|---|
| Amending acts and amended acts | [EUR-Lex](https://eur-lex.europa.eu/) | [e-Gov 法令検索](https://laws.e-gov.go.jp/) |
| Amendment rationale | COM documents (European Commission proposals) accompanying each amending act | The "Reasons" section of the bill text accompanying each amending act |
| Bill-to-act linkage | (directly available via EUR-Lex amendment metadata) | [Japanese Laws and Regulations Index](https://hourei.ndl.go.jp/) is used to look up the bill for each amending act, and the bill text is then retrieved from the [House of Representatives' bill page](https://www.shugiin.go.jp/) |

The English version of EUR-Lex is used for the EU subset, because it has
the fewest missing-content records across the legislative-act series.

## 3. Collection Window

| Jurisdiction | Window |
|---|---|
| EU | 2010 – 2025 |
| JA | 2019 – 2025 |

Pre-2010 EU legislative acts have sparser cross-act reference metadata
and inconsistent rationale formatting, which makes the
amending-act → COM document → amended-act chain difficult to follow
reproducibly. The Japanese window is set so the Japanese corpus is
comparable in size to the EU corpus.

## 4. Amendment Rationale Collection

### 4.1 EU rationale

For each amending act collected from EUR-Lex's annual Legislative Acts
indexes within the window, we retrieve the corresponding **COM document**
(the originating European Commission proposal) in HTML form, then
extract the **explanatory memorandum**.

The explanatory memorandum is structured with sectional headings whose
exact wording varies across documents. Extraction is restricted to
sections that directly state the proposal's motivation and objectives.
Representative inclusion targets:

- *Reasons for and objectives of the proposal*
- *General context*

Sections that focus on contextual or compliance information rather than
motivation are excluded, e.g.:

- *Existing provisions in the area of the proposal*
- *Consistency with existing policy provisions in the policy area*

Because section titles drift across the 16-year window and a single
rule-based extraction does not stably recover the motivational portion,
extraction is performed by hand under a fixed definition of what counts
as a rationale.

Yield: **340** EU rationales (one per admissible amending act).

### 4.2 JA rationale

For each amending act collected from e-Gov within the window, we use the
e-Gov API to collect the act's metadata. Using the Japanese Laws and
Regulations Index, we look up the bill (議案) corresponding to each
amending act as submitted to the House of Representatives. From the
Shugiin bill page we collect the full bill text, and from that text we
extract the section labelled **「理由」 (Reasons)** as the rationale.

Yield: **363** JA rationales.

## 5. Amended Acts and Pre-/Post-Revision Pairing

### 5.1 EU

Each EU consolidated act on EUR-Lex is published as a series of dated
versions, identified by a versioned CELEX-derived ID of the form
`32013R1308-20140101` (act, then snapshot date). For each amending act
`α`, we trace the amendment relationships in EUR-Lex to identify:

- the amended act(s) `β` it modifies,
- the version of `β` immediately *before* `α` took effect, and
- the version of `β` immediately *after* `α` took effect.

The pre- and post-revision versions of each `β` are paired to obtain the
revision-pair for that amendment event.

### 5.2 JA

For each amending act in e-Gov, the metadata identifies the amended
act(s). The law history of each amended act is retrieved, and the
versions immediately before and after the amending act's effective date
are located by law-history ID.

Amended acts are retained only when:

1. their pre- and post-revision versions can be uniquely identified, and
2. the article body of both versions is retrievable via the e-Gov API.

## 6. Article-Level Alignment

The amendment metadata published on EUR-Lex and e-Gov identifies the
amended acts but does not provide article-level correspondences between
pre- and post-revision versions. Article-level alignment is therefore
performed from the text.

### 6.1 Article representation

Each version of an amended act is split into individual articles. Each
article is represented as a triple of three fields:

1. **Article number** (`article_number`),
2. **Caption** (`caption`) — the article's heading or title,
3. **Body** (`text`).

### 6.2 Filtering unchanged articles

Articles whose three fields are all identical between the pre- and
post-revision versions are removed (no actual change).

Articles whose **body text is identical** but for which only the article
number or caption changed are also removed. These edits correspond to
renumbering or relabelling without substantive content change.

### 6.3 Alignment by caption + body similarity

For the remaining articles in `β.before` and `β.after`, alignment is
attempted with the following priority:

1. **Exact caption match.** If the caption strings are identical, the
   pair is accepted as a one-to-one alignment.

2. **Dice coefficient on body text.** For caption mismatches, the body
   text is tokenized:
   - **EU**: word-level whitespace + punctuation tokens (English).
   - **JA**: character-level tokens.

   The Dice coefficient is computed between the token sets of every
   candidate pair; pairs are enumerated in descending Dice order, and
   pairs with **Dice ≥ 0.7** are greedily accepted in one-to-one
   fashion.

3. **Simpson coefficient fallback.** For pairs that fall below the Dice
   threshold but where one version still contains a substantial portion
   of the other (typical of edits that delete or add large sections),
   the Simpson coefficient is also computed. A pair is accepted as a
   candidate when:
   - Simpson coefficient `≥ 0.95`, and
   - the article body is `> 40 characters` in length (to suppress
     spurious matches on very short stubs).

4. **Article-number fallback.** When both Dice and Simpson fall below
   their thresholds, the article-number alone is examined. This fallback
   is applied only when no article-number movement is detected anywhere
   within the same amending act `α`. Otherwise number matches in a
   renumbered act would inject false alignments.

The thresholds and tie-breaking rules were tuned by manual inspection of
misalignments, prioritising the avoidance of incorrect alignments over
recall.

### 6.4 Filtering minor edits

Aligned pairs that correspond to non-substantive edits are discarded:

- **Regex-detected minor edits** — typographical corrections,
  cross-reference renumbering (e.g. updating "Article 15" to
  "Article 16" after a structural shift in another part of the act),
  and similar surface-level changes.
- **Near-identical body text** — pairs with **Dice ≥ 0.99** on the body
  text are treated as minor and discarded.

### 6.5 Newly added articles

Articles introduced for the first time by `α` (no pre-revision version)
are excluded; they cannot participate in either Rat2Rev or Rev2Rev,
which both require a pre-revision article.

### 6.6 Target articles

The pre-revision versions of all article pairs that:

- aligned via caption / Dice / Simpson / article-number fallback (§6.3),
- survived minor-edit filtering (§6.4), and
- have a pre-revision version (§6.5)

constitute the **target articles** `A_e` of amendment event `e`.

## 7. Article-Level Corpus

### 7.1 Version selection

For each amended act, the version **immediately before its earliest
amendment included in the collection window** is used. For acts that
have no amendments included in the window, the version in force at the
**start of the window** is used (2010 for EU, 2019 for JA). This gives
each article exactly one version in `D`.

If an article is amended multiple times within the window, only the
first amendment contributes to qrels; the second amendment's
pre-revision version is the post-revision version of the first
amendment, which differs from the corpus version of the article.

### 7.2 Non-amended acts

To form a realistic candidate pool, articles from acts that were **not**
amended within the window are also included in the corpus. For each such
act, the version in force at the start of the window is included.

## 8. Query and Qrel Derivation

Given §6's target articles `A_e` for every amendment event in the
collection window, queries and qrels are derived deterministically.

### 8.1 Rat2Rev

- **Queries.** One query per amendment event, with text = `r_e` (the
  rationale extracted in §4).
- **Qrels.** `(r_e, a)` is a positive iff `a ∈ A_e`. Binary judgments
  (`score = 1` for positives, omitted otherwise).

### 8.2 Rev2Rev

- **Queries.** For each event with `|A_e| ≥ 2`, sample
  `min(|A_e|, 5)` distinct query-anchor articles uniformly at random.
  The query for each anchor `a` consists of its pre-revision text `a`
  and its post-revision text `ã`.
- **Qrels.** `((a, ã), a')` is a positive iff `a' ∈ A_e \ {a}`. The
  query article `a` itself is excluded from the relevant set; at
  retrieval time the post-filter `qid ≠ docid` is applied to handle
  the case where `a` appears in the run.

The cap at five queries per event prevents a single high-fanout
amendment event from dominating per-query averaged metrics.

## 9. Train / Validation / Test Split

Splits are constructed at **amendment-event granularity**, not query
granularity. All queries derived from the same amendment event — both
Rat2Rev and Rev2Rev — are assigned to the **same** split, so that a
model trained on a Rev2Rev query from event `e` does not also see the
Rat2Rev query from `e` in evaluation.

Each language's set of amendment events is partitioned into approximately
equal thirds:

| | EU | JA |
|---|---:|---:|
| Total events | 340 | 363 |
| Train events | 113 | 121 |
| Validation events | 114 | 121 |
| Test events | 113 | 121 |

Cross-split overlap of amendment events is zero in all (task, lang,
split-pair) combinations. EU and JA event sets are disjoint by
construction (independently sourced).
