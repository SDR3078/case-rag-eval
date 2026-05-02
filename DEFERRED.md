# Deferred work — to be done once `ANTHROPIC_API_KEY` is set

Phase 4a (the retrieval portion of the matrix) is complete and documented in
[`evals/RESULTS.md`](evals/RESULTS.md). The remaining work is **API-gated**:
generation, LLM-as-judge faithfulness/correctness, ROUGE-L baseline, refusal
precision/recall, and `tau_low` calibration all need the Anthropic API.

This file is a step-by-step resume plan plus a clean cost estimate.

## Prerequisites

```bash
export ANTHROPIC_API_KEY=sk-ant-...
# Optional, only if comparing Voyage embeddings:
export VOYAGE_API_KEY=pa-...
```

Verify reachability:

```bash
.venv/bin/python -c "import os; assert os.environ.get('ANTHROPIC_API_KEY'), 'missing ANTHROPIC_API_KEY'"
```

## Phase 4b — generation + judge + refusal

Goal: complete the experimentation matrix.  Retrieval (hit@k, MRR) is done;
generation is not.

1. **Pick the top-3 retrieval configs** for the generation sweep. Read
   `evals/RESULTS.md` §3 — current top-3 by hit@5_any are `max`, `rerank`,
   `embed_bge_large` (and `default`/`retrieval_hybrid` tie behind). Cross with
   the 3 prompt variants (`prompts/system_v{1,2,3}.md`) → 9 generation cells.
2. **Run generation** on the 64 in-scope cases per cell: ~580 Anthropic calls.
   Cost ≈ $7. Sonnet 4.6 with prompt caching enabled (system prompt cached).
3. **LLM-as-judge** for faithfulness + correctness using `claude-sonnet-4-6`
   as the judge. One judge call per generated answer ≈ $7.
4. **ROUGE-L baseline** vs the canonical FAQ answer text — purely local, no
   API call.
5. **Refusal precision/recall** on the 8 OOS cases plus a `tau_low` sweep
   ({0.20, 0.25, ..., 0.50}) per embedding model — architecture §8 calibration.
   Cost ≈ $1.
6. **Update `evals/RESULTS.md`** §8 (generation) with the populated table
   plus one paragraph of commentary per metric.

Total expected cost: **~$15 in API**, plus ~30 min wall clock.

The runner currently raises `NotImplementedError` for the `--full` branch;
implement the generation + judge code paths there. The retrieval-half results
already in `evals/results/{config}/raw.jsonl` should be reused (no need to
re-retrieve).

## Phase 4b (optional) — Voyage embedding variant

If `VOYAGE_API_KEY` is set, drop a YAML in `experiments/` modelled on
`bge_large_hybrid_rerank.yaml` but with `embedding.backend: voyage_3_large`.
Build the index (`python -m ingest --config experiments/voyage_3_large.yaml`)
and re-run the embedding axis with Voyage as a third data point. Document in
`RESULTS.md §4` whether the API model meaningfully beats local bge-large.

If skipped: note in `RESULTS.md §1` that the embedding axis was reported on
local models only.

## Phase 7b — final wrap-up

Once Phase 4b lands, refresh `README.md` so the design narrative cites the
full numbers (not just retrieval). Also re-run a fresh-clone sanity check:

```bash
git clean -fdx index/ evals/results/
pip install -r requirements.txt
export ANTHROPIC_API_KEY=...
python -m ingest --config experiments/default.yaml
python -m evals.run --all --full
python -m app "test question"
```

Confirm everything runs end-to-end without manual intervention.

