# Experiment matrix results

## 1. Overview

This file reports the experiment matrix from `docs/01_architecture.md`. Retrieval metrics — hit@1 / hit@3 / hit@5 (any-match), hit@5 (all-match for multi-FAQ cases), MRR, build-time, per-query latency — are populated for every config in §3. Generation metrics — LLM-judge faithfulness + correctness, ROUGE-L baseline, refusal precision/recall, `tau_low` calibration — populate §5 only for configs run with `--full` (generation requires `OPENAI_API_KEY`).

**Generation status:** at least one config has been run with `--full`; per-config numbers appear in §5. Configs without generation data are listed with TBD.

Voyage AI was excluded from the embedding axis because `VOYAGE_API_KEY` is not set; bge-small vs. bge-large still gives us a clean within-family comparison. Voyage can be added later by dropping a YAML into `experiments/` (copy `max.yaml` and switch `embedding.backend` to `voyage_3_large`).

## 2. Configs run

| config | chunking | embedding | retrieval | reranker |
|---|---|---|---|---|
| `chunking_section` | per_question_with_section | bge_small | dense | none |
| `chunking_section_split_long` | per_question_with_section_split_long | bge_small | dense | none |
| `default` | per_question | bge_small | dense | none |
| `default_v2` | per_question | bge_small | dense | none |
| `default_v3` | per_question | bge_small | dense | none |
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
| `default` | 0.833 | 0.931 | 0.958 | 0.917 | 0.883 | 72 | 0.0 | 23.9 |
| `default_v2` | 0.833 | 0.931 | 0.958 | 0.917 | 0.883 | 72 | 25.7 | 23.0 |
| `default_v3` | 0.833 | 0.931 | 0.958 | 0.917 | 0.883 | 72 | 25.7 | 20.3 |
| `embed_bge_large` | 0.875 | 0.958 | 0.958 | 0.944 | 0.916 | 72 | 0.0 | 223.6 |
| `max` | 0.833 | 0.972 | 0.972 | 0.944 | 0.896 | 72 | 0.0 | 102639.2 |
| `rerank` | 0.833 | 0.944 | 0.972 | 0.931 | 0.894 | 72 | 0.0 | 118912.1 |
| `retrieval_bm25` | 0.778 | 0.861 | 0.917 | 0.819 | 0.837 | 72 | 30.9 | 1.2 |
| `retrieval_hybrid` | 0.847 | 0.917 | 0.958 | 0.903 | 0.893 | 72 | 0.0 | 24.1 |

Notes: `hit@5 (any)` requires at least one expected chunk in the top-5 - this is the headline metric. `hit@5 (all)` requires every expected chunk in the top-5 and is dominated by the 16 multi-FAQ cases (single-target cases are equivalent to `hit@5 (any)`). `build_s` is wall-clock for an actual index build at this run; 0.0 means the index was cached from a prior run.

## 4. Adversarial breakdown

Hit@5 (any) restricted to the 8 adversarial cases. These are the near-duplicate / lexical-trap / boundary-numeric cases that stress disambiguation.

| config | adversarial hit@5 (any) | overall hit@5 (any) | gap |
|---|---:|---:|---:|
| `chunking_section` | 0.625 | 0.917 | -0.292 |
| `chunking_section_split_long` | 0.625 | 0.917 | -0.292 |
| `default` | 0.750 | 0.958 | -0.208 |
| `default_v2` | 0.750 | 0.958 | -0.208 |
| `default_v3` | 0.750 | 0.958 | -0.208 |
| `embed_bge_large` | 0.750 | 0.958 | -0.208 |
| `max` | 0.750 | 0.972 | -0.222 |
| `rerank` | 0.750 | 0.972 | -0.222 |
| `retrieval_bm25` | 0.750 | 0.917 | -0.167 |
| `retrieval_hybrid` | 0.750 | 0.958 | -0.208 |

## 5. Generation results

Per-config generation metrics (populated by `--full`). Faithfulness and correctness are 0–5 LLM-judge scores against the expected answer themes; ROUGE-L is a string-overlap baseline against the canonical FAQ chunk text(s). Refusal P/R is computed over the 80-case set (positive class = OOS that the system correctly refused). The judge defaults to the same model as the generator unless `judge.model` is set in the YAML — note the self-evaluation bias when both columns are filled by the same model.

| config | prompt | generator | judge | faithfulness | correctness | ROUGE-L | refusal P | refusal R | refusal F1 |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| `default` | `system_v1` | `gpt-4o-mini` | `gpt-4o-mini` | 4.77 | 4.63 | 0.308 | 0.727 | 1.000 | 0.842 |
| `default_v2` | `system_v2` | `gpt-4o-mini` | `gpt-4o-mini` | 4.78 | 4.59 | 0.298 | 0.727 | 1.000 | 0.842 |
| `default_v3` | `system_v3` | `gpt-4o-mini` | `gpt-4o-mini` | 4.65 | 4.49 | 0.316 | 0.615 | 1.000 | 0.762 |
| `embed_bge_large` | `system_v1` | `gpt-4o-mini` | `gpt-4o-mini` | 4.65 | 4.51 | 0.293 | 0.615 | 1.000 | 0.762 |
| `max` | `system_v1` | `gpt-4o-mini` | `gpt-4o-mini` | 4.79 | 4.68 | 0.306 | 0.727 | 1.000 | 0.842 |
| `rerank` | `system_v1` | `gpt-4o-mini` | `gpt-4o-mini` | 4.72 | 4.61 | 0.306 | 0.667 | 1.000 | 0.800 |
| `retrieval_hybrid` | `system_v1` | `gpt-4o-mini` | `gpt-4o-mini` | 4.78 | 4.61 | 0.313 | 0.727 | 1.000 | 0.842 |

### Refusal floor calibration — `default`

Floor-only refusal precision/recall at each `tau_low`, computed from logged top-1 dense scores. The configured value lives in the YAML; this sweep helps pick a calibrated knee.

| tau_low | floor TP | floor FP | floor FN | precision | recall |
|---:|---:|---:|---:|---:|---:|
| 0.20 | 0 | 0 | 8 | - | 0.000 |
| 0.25 | 0 | 0 | 8 | - | 0.000 |
| 0.30 | 0 | 0 | 8 | - | 0.000 |
| 0.35 | 0 | 0 | 8 | - | 0.000 |
| 0.40 | 0 | 0 | 8 | - | 0.000 |
| 0.45 | 0 | 0 | 8 | - | 0.000 |
| 0.50 | 1 | 0 | 7 | 1.000 | 0.125 |
