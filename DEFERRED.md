# Deferred work — what remains for follow-up runs

**Phase 4a and Phase 4b are both complete.** [`evals/RESULTS.md`](evals/RESULTS.md)
contains the full retrieval matrix (10 configs) and the populated generation
table (7 configs run with `--full` through `gpt-4o-mini` on OpenAI proper,
gpt-4o-mini also as the LLM-as-judge). What's documented here is **rerunnable
follow-up work** — both for re-running Phase 4b on a different provider /
model, and for the optional Voyage embedding variant.

The backend is **OpenAI-compatible** (see `generator.py` and `evals/judge.py`).
Same code talks to OpenAI, Azure, OpenRouter, Together, Groq, LM Studio,
Ollama, vLLM, etc. — pick whichever you have credit / hardware for and set
`OPENAI_BASE_URL` (or `generation.base_url` in the YAML) accordingly.

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
`llama-3.3-70b-versatile`. For other providers, copy and tweak the two fields
under `generation:`.

## Re-running Phase 4b on a different provider/model

```bash
# One config end-to-end (~6 min for 80 cases × 2 calls on OpenAI gpt-4o-mini):
python -m evals.run --config experiments/default.yaml --full

# All configs end-to-end:
python -m evals.run --all --full
```

What `--full` does per case:
1. **Retrieval** as in Phase 4a (already cached for the existing configs).
2. **Generation** — calls the configured `generation.model` through the
   `Generator` class.
3. **Refusal-floor short-circuit** — if dense top-1 < `tau_low`, skip
   generation and emit the canonical refusal.
4. **LLM-as-judge** (in-scope cases only) — separate call to score
   faithfulness + correctness on a 0–5 scale against `expected_answer_themes`.
5. **ROUGE-L** — local computation against the canonical FAQ chunk text(s).
6. **Refusal flags** — both floor-driven and model-driven; aggregated into
   precision/recall/F1 over all 80 cases.
7. **`tau_low` sweep** — analytical from logged top-1 scores, sweeps
   {0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50}.

## Cost reality on free / paid tiers

What the actual matrix run consumed (7 configs, 80 cases each):
- **OpenAI `gpt-4o-mini`** (the configs in `experiments/`): ≈ $0.10 / config →
  **< $1 total** for the matrix. ~6 min wall clock per fast config (no
  reranker), ~50 min per `rerank` config (CPU cross-encoder), ~2.5 hours
  per `max` config (CPU cross-encoder + bge-large + section_split).
- **Groq `llama-3.3-70b-versatile` free tier**: hit the **100 k TPD daily
  cap** in one config run (the eval needs ~450 k tokens). Free Groq is not
  practical for the full matrix without paying for the Dev tier or
  switching to a smaller model with separate quota
  (`llama-3.1-8b-instant`).

## Judge bias caveat (most consequential follow-up)

By default the judge uses the same model as the generator (no `judge.model`
override needed in the YAML). This is the biggest known limitation of the
shipped numbers — a model is rarely a strict critic of its own output. The
faith/corr deltas between top configs (max vs default ≈ 0.02–0.05) are
within self-evaluation noise. To re-run with a stronger / different judge,
add to the YAML:

```yaml
judge:
  model: gpt-4o                # or any stronger model your provider exposes
  base_url: https://api.openai.com/v1   # or omit if same as generation
```

## Voyage embedding variant (optional)

If `VOYAGE_API_KEY` is set, drop a YAML in `experiments/` modelled on
`bge_large_hybrid_rerank.yaml` but with `embedding.backend: voyage_3_large`.
Build the index, then re-run the embedding axis with Voyage as a third
data point.

The current matrix shows bge-large already underperforms bge-small on
generation faith/corr despite better hit@1 (broader retrieval = more
ambiguous context). Voyage is unlikely to break this pattern but it's the
only outstanding embedding axis variant.

## Fresh-clone sanity check

```bash
git clean -fdx index/ evals/results/
pip install -r requirements.txt
export OPENAI_API_KEY=...
python -m ingest --config experiments/default.yaml
python -m evals.run --all --full
python -m app "test question"
```

Confirm everything runs end-to-end without manual intervention.

