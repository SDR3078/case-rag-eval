# Evaluation set — `cases.jsonl`

The eval set drives the eval pipeline's metrics: retrieval (hit@k, MRR, recall@k),
generation (LLM-as-judge faithfulness/correctness + ROUGE-L baseline) and
refusal precision/recall. This README documents what is in the file, how the
80 cases were sourced, and how to extend it.

## What's in the file

`cases.jsonl` is one JSON object per line with this schema:

| field | type | description |
|---|---|---|
| `id` | string | Stable identifier prefixed by category (`g_`, `m_`, `a_`, `o_`) |
| `category` | enum | `golden_path` / `multi_faq` / `adversarial` / `out_of_scope` |
| `question` | string | The user's question, paraphrased so it does not echo the FAQ verbatim |
| `expected_chunk_ids` | list of strings | Chunk IDs from `index/default/chunks.jsonl` that contain the answer; empty for OOS |
| `expected_answer_themes` | list of strings | 3–8 short phrases the generated answer must touch (the LLM-as-judge metric scores against these). Empty for OOS |
| `expected_behaviour` | enum | `answer` (system should produce a substantive reply) or `refuse` |
| `notes` | string | Why this case exists and what it stresses |

`validate_cases.py` enforces the schema and checks every `expected_chunk_ids`
entry exists in `index/default/chunks.jsonl`. Run it before declaring a change
done — the eval pipeline assumes these invariants.

## Distribution actually shipped

Total: **80 cases**, in line with the 50–100 band the brief specifies and the
15–20 OOS subset called for in `docs/01_architecture.md` §8.

| category | count | share |
|---|---:|---:|
| `golden_path` | 48 | 60 % |
| `multi_faq` | 16 | 20 % |
| `adversarial` | 8 | 10 % |
| `out_of_scope` | 8 | 10 % |

### Golden-path coverage by source section

The 48 golden-path cases sample real `### Question` entries proportionally to
the size of each top-level `## Section`. Every section is represented at
roughly its share of the corpus, so per-section retrieval quality can be read
out of any aggregate metric.

| Section | Section size (chunks) | Corpus share | Eval cases | Eval share |
|---|---:|---:|---:|---:|
| Climate Delegated Act | 195 | 59.3 % | 28 | 58.3 % |
| Taxonomy-Alignment Reporting | 37 | 11.2 % | 5 | 10.4 % |
| Taxonomy-Eligibility reporting (part 2) | 33 | 10.0 % | 5 | 10.4 % |
| EU Taxonomy - General | 23 | 7.0 % | 4 | 8.3 % |
| Taxonomy-Eligibility reporting (part 1) | 22 | 6.7 % | 3 | 6.2 % |
| Disclosures Delegated Act - General | 11 | 3.3 % | 2 | 4.2 % |
| Complementary Climate Delegated Act | 8 | 2.4 % | 1 | 2.1 % |
| **Total** | **329** | **100 %** | **48** | **100 %** |

Multi-FAQ cases pull from chunks across all 7 sections, so the actual
chunk-touched coverage is broader than the table above suggests.

## How cases were sourced

Each category has an explicit sourcing recipe so a reviewer can reproduce or
extend the set without ambiguity.

### Golden-path (48)
- One `### Question` chunk per case.
- Question text paraphrased — surface form deliberately changed: tense flip,
  formal-to-colloquial register, synonym swap (`shall` → `must` → `is required to`),
  switch from third- to first-person where natural, and reformulation
  (`Can X?` → `Is X allowed?`). Questions never echo the FAQ heading verbatim.
- Themes (3–8 phrases) are pulled from the answer body. They are paraphrased
  too, so the LLM-as-judge does not collapse into a string-match check.
- Sample spans all 7 top-level sections in roughly the proportions in the
  table above. Within Climate Delegated Act we deliberately covered a wide
  range of activities — forestry, manufacturing, transport, buildings,
  hydrogen, climate adaptation — rather than oversampling any one.

### Multi-FAQ synthesis (16)
- 2–3 chunks each, chosen because the question genuinely needs both/all to be
  answered. Examples:
  - *m_001*: "what to report" + "what if I have no eligible activity" → both
    chunks needed for a full reply.
  - *m_007*: three chunks on climate-risk methodology (minimum scope, IPCC
    pathway selection, IPCC version) — each addresses a different
    sub-question that a real user typically asks together.
  - *m_011*: green non-EU debt + green sovereign debt — eligibility rules
    differ between the two and the answer requires reconciling both FAQs.
- For the metric: a multi-FAQ case is a recall@k stress test (the system must
  surface *all* required chunks in its top-k, not just one), and a faithfulness
  stress test (the answer must integrate them coherently rather than copy one).

### Adversarial (8)
Designed to expose retrieval brittleness:

- **Near-duplicates across sections.** The CCDA section duplicates 8 questions
  verbatim from the CDA section. *a_001* asks about nuclear waste; the
  identical question text exists at `faq_2adb2cb557` (CDA) and
  `faq_ba47d5704a` (CCDA). The case marks the CCDA chunk as correct because
  the question's framing ("transitional activities") is the CCDA angle. *a_006*
  marks **both** duplicate chunks as acceptable because the question is
  genuinely identical between the two — this tests that retrieval doesn't
  silently split scores across true duplicates.
- **Lexical-overlap traps.** *a_005* asks about "Article 8 disclosures" in the
  context of iron-and-steel manufacturing; "Article 8" appears in dozens of
  FAQs, but only `faq_68befa8346` (TSC compliance for activities outside the
  EU) actually answers the question.
- **Semantic vs. surface form.** *a_002* and *a_004* test cases where two FAQs
  differ subtly (eligibility-side vs. alignment-side derivatives treatment;
  CSRD scope FAQ vs. NFRD interaction FAQ).
- **Boundary numerics.** *a_003* uses the exact 13 ha threshold from the
  forestry climate-benefit-analysis exemption.

### Out-of-scope (8)
Three sub-types from the architecture's §8 evaluation plan:

- **(a) Clearly unrelated** — *o_001* (weather), *o_002* (recipe), *o_003*
  (sports). 3 cases.
- **(b) Adjacent but uncovered** — *o_004* (Belgian income taxes), *o_005*
  (EU ETS carbon price), *o_006* (CSDDD due-diligence directive). 3 cases. We
  verified by `grep` that none of these topics appear in
  `docs/taxonomy_faqs_cleaned.md`.
- **(c) In-domain but uncovered angle** — *o_007* (penalties for missing
  Article 8 disclosures), *o_008* (appeals against external-verifier
  determinations). 2 cases. The corpus discusses the obligation and the
  verification but has no FAQ on enforcement or appeals.

Each OOS case has `expected_behaviour: "refuse"` and empty `expected_chunk_ids`
/ `expected_answer_themes`. Refusal precision/recall is computed over this
subset.

## How to add a new case

1. Pick a chunk ID from `index/default/chunks.jsonl` (or chunk IDs, for
   multi-FAQ).
2. Compose a question that is a non-trivial paraphrase of the FAQ heading —
   change tense, register, perspective, or synonyms.
3. Distil 3–8 themes from the answer body. Phrase them as short noun phrases
   or short clauses, not exact quotes (the eval is a coverage check, not
   string-match).
4. Append a JSON line to `cases.jsonl`. Choose the next free `id` in the
   relevant category prefix (`g_`, `m_`, `a_`, `o_`).
5. Run `.venv/bin/python evals/validate_cases.py` to confirm the chunk IDs
   exist and the schema is intact.
6. (Optional, recommended) Spot-check that the question is genuinely answered
   by the chunk(s) you cited — paraphrasing can drift, especially for
   adversarial cases.

## Known limitations

- **Paraphrasing was Claude-assisted, not crowdsourced.** Real users phrase
  questions worse than this — typos, abbreviations, run-on sentences, mixed
  languages. The eval set is a *clean* approximation; numbers here will be
  upper bounds versus production traffic.
- **Multi-FAQ synthesis cases reflect what the corpus permits, not what real
  users would ask.** We picked FAQ pairs that genuinely complement, but real
  users sometimes ask questions that bridge an FAQ with information that
  isn't in any single entry. Those are inherently OOS for this corpus.
- **Adversarial difficulty is modest.** Eight cases is enough to show whether
  retrieval handles obvious traps; it is not enough to characterise the
  failure surface. If the matrix shows the adversarial bucket lagging the
  golden-path bucket by a wide margin, the next step is to grow this bucket
  to ~16 cases and stratify by trap type.
- **OOS adjacency is approximate.** "Adjacent but uncovered" is judgement.
  Some reviewers might argue that *o_005* (EU ETS) is sufficiently
  EU-climate-policy-adjacent that a charitable system should retrieve and
  decline rather than refuse outright; the cases shipped here treat that as a
  refusal because the refusal eval needs a binary label. This is a known scoring edge.
- **No multilingual coverage.** All 80 cases are English. The corpus
  occasionally quotes French/German legal terms but the FAQ prose is English,
  and the take-home brief does not require multilingual support.
- **Themes are not exhaustive.** They are sufficient-conditions for a
  faithful answer, not necessary-conditions; a generated answer may legitimately
  add more material than the themes capture (e.g. a citation block).

## Files

- `cases.jsonl` — 80 cases, one JSON per line.
- `validate_cases.py` — schema and chunk-id integrity check; exits non-zero
  on any failure. Run before commits that touch this directory.
- `RESULTS.md` — populated by the eval runner.
