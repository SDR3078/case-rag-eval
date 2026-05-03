# Deferred work — to be done once `OPENAI_API_KEY` is set

Phase 4a (retrieval matrix) is complete and documented in
[`evals/RESULTS.md`](evals/RESULTS.md). **Phase 4b is now implemented** —
generation, LLM-as-judge faithfulness/correctness, ROUGE-L baseline, refusal
precision/recall, and `tau_low` calibration all run when `--full` is passed
to the eval runner. The remaining work is **execution** (and reading the
results), not coding.

The backend is **OpenAI-compatible** (see `generator.py`) and so is the
judge (`evals/judge.py`). Same code talks to OpenAI, Azure, OpenRouter,
Together, Groq, LM Studio, Ollama, vLLM, etc. — pick whichever you have
credit / hardware for and set `OPENAI_BASE_URL` (or `generation.base_url`
in the YAML) accordingly.

## Prerequisites

```bash
export OPENAI_API_KEY=sk-...
# Optional: any OpenAI-compatible endpoint. Default = OpenAI proper.
# export OPENAI_BASE_URL=https://api.groq.com/openai/v1     # Groq
# export OPENAI_BASE_URL=https://openrouter.ai/api/v1        # OpenRouter
# export OPENAI_BASE_URL=http://localhost:1234/v1            # LM Studio
# export OPENAI_BASE_URL=http://localhost:11434/v1           # Ollama
```

`experiments/groq.yaml` is a worked example pre-pinning Groq's URL +
`llama-3.3-70b-versatile`. For other providers, copy and tweak the two
fields under `generation:`.

## Phase 4b — generation + judge + refusal

Goal: populate §8 of `evals/RESULTS.md`.

```bash
# One config end-to-end (~3 min on Groq's free tier for 80 cases × 2 calls
# each = ~160 calls):
python -m evals.run --config experiments/groq.yaml --full

# All 9 configs end-to-end (~25 min):
python -m evals.run --all --full
```

What `--full` does per case:
1. **Retrieval** as in Phase 4a (already cached for these configs).
2. **Generation** — calls the configured `generation.model` through the
   `Generator` class.
3. **Refusal-floor short-circuit** — if dense top-1 < `tau_low`, skip
   generation and emit the canonical refusal.
4. **LLM-as-judge** (in-scope cases only) — separate call to score
   faithfulness + correctness on a 0-5 scale against `expected_answer_themes`.
5. **ROUGE-L** — local computation against the canonical FAQ chunk text(s).
6. **Refusal flags** — both floor-driven and model-driven; aggregated into
   precision/recall/F1 over all 80 cases.
7. **`tau_low` sweep** — analytical from logged top-1 scores, sweeps
   {0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50}.

### Cost estimate (gpt-4o-mini default)

- Generation: 80 cases × ~3k tokens = 240k input + ~25k output tokens.
- Judge: 64 in-scope cases × ~3.5k tokens = 224k input + ~10k output tokens.
- Per config: ~$0.07 input + ~$0.02 output ≈ **$0.09**.
- 9 configs end-to-end: **<$1**.

### Cost on Groq (free tier)

Groq's free tier with `llama-3.3-70b-versatile` is rate-limited (≈30 RPM)
but free; the eval pace is bounded by the rate limit, not cost. Expect
~3 min per config of wall clock.

### Judge bias caveat

By default the judge uses the same model as the generator (no `judge.model`
override needed in the YAML). This is biased — a model is rarely a strict
critic of its own output. To run an unbiased pass, add to the YAML:

```yaml
judge:
  model: gpt-4o                # or any stronger model your provider exposes
  base_url: https://api.openai.com/v1   # or omit if same as generation
```

## Phase 4b (optional) — Voyage embedding variant

If `VOYAGE_API_KEY` is set, drop a YAML in `experiments/` modelled on
`bge_large_hybrid_rerank.yaml` but with `embedding.backend: voyage_3_large`.
Build the index, then re-run the embedding axis with Voyage as a third
data point.

If skipped: note in `RESULTS.md §1` that the embedding axis was reported on
local models only.

## Phase 7b — final wrap-up

Once Phase 4b lands, refresh `README.md` so the design narrative cites the
generation numbers (not just retrieval). Re-run the fresh-clone sanity
check:

```bash
git clean -fdx index/ evals/results/
pip install -r requirements.txt
export OPENAI_API_KEY=...
python -m ingest --config experiments/default.yaml
python -m evals.run --all --full
python -m app "test question"
```

Confirm everything runs end-to-end without manual intervention.

