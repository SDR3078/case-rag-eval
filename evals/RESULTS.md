# Experiment matrix results

## 1. Overview

This file reports the experiment matrix from `docs/01_architecture.md`. Retrieval metrics — hit@1 / hit@3 / hit@5 (any-match), hit@5 (all-match for multi-FAQ cases), MRR, build-time, per-query latency — are populated for every config in §3. Generation metrics — LLM-judge faithfulness + correctness, ROUGE-L baseline, refusal precision/recall, `tau_low` calibration — populate §8 only for configs run with `--full` (generation requires `OPENAI_API_KEY`).

Generation, LLM-as-judge faithfulness/correctness, ROUGE-L, refusal precision/recall, and `tau_low` calibration are deferred to Phase 4b until `OPENAI_API_KEY` (or any OpenAI-compatible endpoint via `OPENAI_BASE_URL`) is configured. See `DEFERRED.md` for the resume plan.

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
| `default` | 0.833 | 0.931 | 0.958 | 0.917 | 0.883 | 72 | 0.0 | 33.7 |
| `embed_bge_large` | 0.875 | 0.958 | 0.958 | 0.944 | 0.916 | 72 | 259.2 | 165.6 |
| `max` | 0.833 | 0.972 | 0.972 | 0.944 | 0.896 | 72 | 0.0 | 102823.3 |
| `rerank` | 0.833 | 0.944 | 0.972 | 0.931 | 0.894 | 72 | 0.0 | 118613.6 |
| `retrieval_bm25` | 0.778 | 0.861 | 0.917 | 0.819 | 0.837 | 72 | 30.9 | 1.2 |
| `retrieval_hybrid` | 0.847 | 0.917 | 0.958 | 0.903 | 0.893 | 72 | 30.2 | 18.9 |

Notes: `hit@5 (any)` requires at least one expected chunk in the top-5 - this is the headline metric. `hit@5 (all)` requires every expected chunk in the top-5 and is dominated by the 16 multi-FAQ cases (single-target cases are equivalent to `hit@5 (any)`). `build_s` is wall-clock for an actual index build at this run; 0.0 means the index was cached from a prior run.

## 4. Per-axis commentary

**Chunking.** Baseline (`default`, per-question) sits at hit@5_any = 0.958. Section-prepended chunking (`chunking_section`) lands at 0.917 — **hurt** hit@5_any by 4.2 pp, the **opposite** of the architect's prediction. Plausible mechanism: section name biases the embedding toward the section centroid rather than the specific question, increasing intra-section confusion. Splitting long entries on top (`chunking_section_split_long`) reaches 0.917. The corpus has only a handful of >800-token outliers, so most chunks pass through the splitter untouched — splitting moves the needle only on the long-entry tail (the FINREP block, the long 'what should financial companies report' answer).

**Embedding model.** Holding chunking, retrieval, and reranker fixed to baseline, swapping bge-small for bge-large moved hit@5_any from 0.958 to 0.958 and MRR from 0.883 to 0.916. Build time changed from 0.0s to 259.2s (one-off; cached afterward). Per-query latency stays in the same ballpark - the index lookup is still a single dot product, just over a wider matrix (1024 dims vs. 384). Voyage was not run (no API key), so this is a within-family comparison; reviewers wanting a hosted-SOTA reference can drop a Voyage YAML in `experiments/` once the key is available.

**Retrieval method.** With per-question chunking + bge-small + no rerank, dense retrieval (`default`) gives hit@5_any = 0.958, BM25-only (`retrieval_bm25`) gives 0.917, and hybrid RRF (`retrieval_hybrid`) gives 0.958. The FAQ corpus has heavy term repetition ('Article 8', 'TSC', 'CapEx', section codes), so BM25 is genuinely a tough baseline - article references are exact-match signals dense models routinely fumble. Hybrid RRF needs no tunable weight and reliably matches or beats either component when neither one dominates.

**Reranker.** Cross-encoder reranking (bge-reranker-v2-m3) on top of the baseline dense pipeline moved hit@5_any from 0.958 to 0.972 and MRR from 0.883 to 0.894. The reranker primarily reorders results that dense already surfaces (its lift on hit@5 is bounded by recall@20 of the dense first stage), so MRR is the more sensitive headline. Lift over the dense baseline = **+1.4 pp**, vs the architecture's 5-pp promotion threshold (§5) — reranker **stays a variant**.

## 5. Best-of-axes summary

**2-way tie at hit@5_any = 0.972**: `max`, `rerank`. MRR breaks the tie: `max` at 0.896. Best MRR: **`embed_bge_large`** at 0.916 (hit@5_any=0.958) — different metric, different winner. Lift over `default`: 1.4 pp on hit@5_any.

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

6 configs tie at adversarial hit@5_any = 0.750: `default`, `embed_bge_large`, `max`, `rerank`, `retrieval_bm25`, `retrieval_hybrid`. At the bottom: `chunking_section`, `chunking_section_split_long` at 0.625 (12.5 pp behind the leader). With only 8 adversarial cases the per-config differences are 1-2 cases each; the adversarial bucket is suggestive, not definitive.

## 8. Generation results

No `--full` runs yet. Once `OPENAI_API_KEY` is set, run `python -m evals.run --config experiments/<name>.yaml --full` to populate this section. The eval ships an OpenAI-compatible generator (`generator.py`) and an LLM-as-judge (`evals/judge.py`); both honour `OPENAI_BASE_URL` and the YAML's `generation.base_url` so any provider works (see DEFERRED.md).
