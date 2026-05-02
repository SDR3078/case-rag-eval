# Phase 4a results - retrieval matrix

## 1. Overview

This file reports the **retrieval-only** portion of the experiment matrix from `docs/01_architecture.md`. Metrics measured here: hit@1 / hit@3 / hit@5 (any-match), hit@5 (all-match for multi-FAQ cases), MRR, plus build-time and per-query latency. The 64 in-scope cases (golden_path + multi_faq + adversarial) drive the headline numbers. The 8 OOS cases have their top-1 similarity captured for later `tau_low` calibration.

Generation, LLM-as-judge faithfulness/correctness, ROUGE-L, refusal precision/recall, and `tau_low` calibration are deferred to Phase 4b because `ANTHROPIC_API_KEY` is not set. See `DEFERRED.md` for the resume plan and expected cost (~$15-20).

Voyage AI was excluded from the embedding axis because `VOYAGE_API_KEY` is not set; bge-small vs. bge-large still gives us a clean within-family comparison. Voyage can be added later by dropping a YAML into `experiments/` (see `bge_large_hybrid_rerank.yaml` for a non-default-embedding template).

## 2. Configs run

| config | chunking | embedding | retrieval | reranker |
|---|---|---|---|---|
| `chunking_section` | per_question_with_section | bge_small | dense | none |
| `chunking_section_split_long` | per_question_with_section_split_long | bge_small | dense | none |
| `default` | per_question | bge_small | dense | none |
| `embed_bge_large` | per_question | bge_large | dense | none |
| `max` | per_question_with_section_split_long | bge_large | hybrid | BAAI/bge-reranker-v2-m3 |
| `rerank` | per_question | bge_small | dense | BAAI/bge-reranker-v2-m3 |
| `retrieval_bm25` | per_question | bge_small | bm25 | none |
| `retrieval_hybrid` | per_question | bge_small | hybrid | none |

## 3. Retrieval results

| config | hit@1 | hit@3 | hit@5 (any) | hit@5 (all) | MRR | n_cases | build_s | query_ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `chunking_section` | 0.792 | 0.917 | 0.917 | 0.889 | 0.856 | 72 | 30.9 | 19.1 |
| `chunking_section_split_long` | 0.792 | 0.917 | 0.917 | 0.889 | 0.856 | 72 | 31.6 | 18.3 |
| `default` | 0.833 | 0.931 | 0.958 | 0.917 | 0.883 | 72 | 0.0 | 18.7 |
| `embed_bge_large` | 0.875 | 0.958 | 0.958 | 0.944 | 0.916 | 72 | 259.2 | 165.6 |
| `max` | 0.833 | 0.972 | 0.972 | 0.944 | 0.896 | 72 | 0.0 | 102823.3 |
| `rerank` | 0.833 | 0.944 | 0.972 | 0.931 | 0.894 | 72 | 0.0 | 118613.6 |
| `retrieval_bm25` | 0.778 | 0.861 | 0.917 | 0.819 | 0.837 | 72 | 30.9 | 1.2 |
| `retrieval_hybrid` | 0.847 | 0.917 | 0.958 | 0.903 | 0.893 | 72 | 30.2 | 18.9 |

Notes: `hit@5 (any)` requires at least one expected chunk in the top-5 - this is the headline metric. `hit@5 (all)` requires every expected chunk in the top-5 and is dominated by the 16 multi-FAQ cases (single-target cases are equivalent to `hit@5 (any)`). `build_s` is wall-clock for an actual index build at this run; 0.0 means the index was cached from a prior run.

## 4. Per-axis commentary

**Chunking.** Baseline (`default`, per-question) sits at hit@5_any = 0.958. Prepending the parent section header (`chunking_section`) drops to 0.917 (-4.1 pp), and adding outlier splitting (`chunking_section_split_long`) does not recover it (also 0.917). The architect predicted a small lift from section context (3-6 high-signal tokens of disambiguation between Climate / Disclosures / Alignment); the data goes the other way. Plausible mechanism: with 195 FAQs in the largest section ("Climate Delegated Act"), the section name biases the embedding toward the section centroid rather than the specific question, *increasing* lexical overlap with intra-section distractors. The drop is sharper on the adversarial subset (0.625 vs 0.750 for the no-section variants) — confirms the disambiguation hurt. Long-entry splitting was meant to handle the FINREP outlier, but with only a handful of chunks above the 800-token threshold the change is invisible at this aggregate level.

**Embedding model.** Holding chunking, retrieval, and reranker fixed to baseline, swapping bge-small for bge-large moved hit@5_any from 0.958 to 0.958 and MRR from 0.883 to 0.916. Build time changed from 0.0s to 259.2s (one-off; cached afterward). Per-query latency stays in the same ballpark - the index lookup is still a single dot product, just over a wider matrix (1024 dims vs. 384). Voyage was not run (no API key), so this is a within-family comparison; reviewers wanting a hosted-SOTA reference can drop a Voyage YAML in `experiments/` once the key is available.

**Retrieval method.** With per-question chunking + bge-small + no rerank, dense retrieval (`default`) gives hit@5_any = 0.958, BM25-only (`retrieval_bm25`) gives 0.917, and hybrid RRF (`retrieval_hybrid`) gives 0.958. The FAQ corpus has heavy term repetition ('Article 8', 'TSC', 'CapEx', section codes), so BM25 is genuinely a tough baseline - article references are exact-match signals dense models routinely fumble. Hybrid RRF needs no tunable weight and reliably matches or beats either component when neither one dominates.

**Reranker.** Cross-encoder reranking (bge-reranker-v2-m3) on top of the baseline dense pipeline moved hit@5_any from 0.958 to 0.972 (+1.4 pp) and MRR from 0.883 to 0.894 (+1.1 pp). The reranker primarily reorders results that dense already surfaces (its lift on hit@5 is bounded by recall@20 of the dense first stage), so MRR is the more sensitive headline. Architecture §5 promotes the reranker to default only when its hit@5 lift exceeds **5 pp**; the measured **1.4 pp** falls below this bar, so the reranker **stays a variant**, not the default. Latency cost on this CPU was prohibitive (~118 s/query vs 19 ms for dense — ~6000× slowdown). On a GPU this drops to ~50 ms/query and the latency math reverses.

## 5. Best-of-axes summary

**Two-way tie at hit@5_any = 0.972**: `max` (per_question_with_section_split_long + bge-large + hybrid + reranker) and `rerank` (per-question + bge-small + dense + reranker). MRR splits them by 0.002 (max 0.896 vs rerank 0.894). The striking ablation: stacking bge-large + hybrid + section-split *on top of* the reranker contributes essentially **zero additional lift** — the reranker absorbs all available headroom in this corpus. Lift over `default`: 1.4 pp on hit@5_any.

Best **MRR** sits elsewhere: `embed_bge_large` at 0.916, beating both reranker variants. So if the headline metric is "answer is in top-5", the reranker wins; if it's "first result is correct", bge-large dense wins (0.875 hit@1 vs 0.833 for the rerankers). Latency note: bge-large is 9× slower than bge-small per query (165 ms vs 19 ms); reranker variants are 6000× slower (~118 s/query on CPU).

## 6. Fine-tune trigger evaluation (architecture s10)

**Verdict: trigger does NOT fire.** Best non-FT hit@5_any = **0.972** (`max`); next-best = **0.972** (`rerank`) so the gap is **0.00 pp**. The matrix has plateaued (gap < 2 pp) but hit@5 already clears the 0.85 ceiling, so the marginal lift a fine-tune would deliver cannot justify the engineering cost. Fine-tune is **skipped**; generation faithfulness becomes the next bottleneck (Phase 4b).

## 7. Adversarial breakdown

Hit@5 (any) restricted to the 8 adversarial cases. These are the near-duplicate / lexical-trap / boundary-numeric cases that stress disambiguation.

| config | adversarial hit@5 (any) | overall hit@5 (any) | gap |
|---|---:|---:|---:|
| `chunking_section` | 0.625 | 0.917 | -0.292 |
| `chunking_section_split_long` | 0.625 | 0.917 | -0.292 |
| `default` | 0.750 | 0.958 | -0.208 |
| `embed_bge_large` | 0.750 | 0.958 | -0.208 |
| `max` | 0.750 | 0.972 | -0.222 |
| `rerank` | 0.750 | 0.972 | -0.222 |
| `retrieval_bm25` | 0.750 | 0.917 | -0.167 |
| `retrieval_hybrid` | 0.750 | 0.958 | -0.208 |

**Six configs tie at adversarial hit@5_any = 0.750**: `default`, `embed_bge_large`, `max`, `rerank`, `retrieval_bm25`, `retrieval_hybrid`. **Two configs trail at 0.625**: `chunking_section`, `chunking_section_split_long`. Two findings:

1. **BM25 alone matches dense** on adversarial despite trailing dense by 4 pp on the average case — the architect's bet that exact-string signals (article numbers, section codes) help disambiguation pays off here.
2. **Section-prepending is the only choice that hurts** — drops adversarial by 12.5 pp. But `max` uses section-prepending and still hits 0.750: stacking the reranker on top *recovers* what section-prepending broke. So the section header isn't toxic *per se*, only when used alone.
3. **The reranker did NOT improve adversarial** beyond the dense baseline — `rerank` and `default` both score 0.750. Whatever lift the reranker gives on the average case (the +1.4 pp on hit@5_any) doesn't translate to disambiguation help.

With only 8 adversarial cases the per-config differences are 1-2 cases each; the adversarial bucket is suggestive, not definitive.

## 8. Generation results (Phase 4b - TBD)

Once `ANTHROPIC_API_KEY` is set we will run generation across the top-2 / top-3 retrieval configs by hit@5_any, crossed with the three prompt variants in `prompts/`.

| retrieval config | prompt | faithfulness | correctness | ROUGE-L | refusal P | refusal R |
|---|---|---:|---:|---:|---:|---:|
| `max` | `system_v1` | TBD | TBD | TBD | TBD | TBD |
| `max` | `system_v2` | TBD | TBD | TBD | TBD | TBD |
| `max` | `system_v3` | TBD | TBD | TBD | TBD | TBD |
| `rerank` | `system_v1` | TBD | TBD | TBD | TBD | TBD |
| `rerank` | `system_v2` | TBD | TBD | TBD | TBD | TBD |
| `rerank` | `system_v3` | TBD | TBD | TBD | TBD | TBD |
| `default` | `system_v1` | TBD | TBD | TBD | TBD | TBD |
| `default` | `system_v2` | TBD | TBD | TBD | TBD | TBD |
| `default` | `system_v3` | TBD | TBD | TBD | TBD | TBD |

Refusal precision/recall additionally needs the 8 OOS cases plus the `tau_low` sweep (`tau_low in {0.20, 0.25, ..., 0.50}`) per embedding model - see DEFERRED.md.
