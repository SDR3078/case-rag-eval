# EU Taxonomy FAQ RAG Chatbot

A single-shot retrieval-augmented Q&A bot over the EU Taxonomy Navigator FAQs
(329 entries across 7 sections). Built for the Sopra Steria NLP take-home.

The 72-hour deadline in the brief was deliberately dropped in favour of a
thorough experimentation matrix and a defensible eval: 80 hand-curated cases ×
**10 retrieval configs** with full retrieval metrics, **7 of which were also
run with `--full`** (generation through `gpt-4o-mini` plus LLM-as-judge
faithfulness + correctness + ROUGE-L + refusal P/R + `tau_low` calibration).
See [`docs/01_architecture.md`](docs/01_architecture.md) for the design
rationale and [`evals/RESULTS.md`](evals/RESULTS.md) for measured results
(retrieval table + generation table + tau_low floor calibration sweep).

## How to read this repo (recommended order)

1. **`docs/01_architecture.md`** — every design choice and *why*. Read this
   first; the rest of the repo executes on it.
2. **`evals/RESULTS.md`** — what the matrix actually showed. Per-axis
   commentary, best-of summary, fine-tune trigger verdict, adversarial breakdown.
3. **`ingest.py`** — chunking strategies and embedding backends (BGE/Voyage),
   with the BGE query/passage prefix asymmetry encoded explicitly. The longest
   module at ~490 lines, but each concern (parsing / chunking / embedding /
   BM25 / persistence) is small and independent.
4. **`retriever.py`** — dense / BM25 / hybrid (RRF k=60) retrieval over a
   persisted index. Small.
5. **`generator.py`** — OpenAI-compatible chat-completions call (`base_url`
   configurable, so any provider works: OpenAI, Azure, OpenRouter, Together,
   Groq, LM Studio, Ollama, vLLM). Prompt-prefix caching is automatic on
   providers that support it. Refusal-floor short-circuit is in `app.py`.
6. **`evals/cases.jsonl`** + `evals/README.md` — 80 hand-curated cases
   (60 % golden, 20 % multi-FAQ synthesis, 10 % adversarial, 10 % out-of-scope)
   with sourcing methodology.

Total LOC excluding prompts and evals: ~1000.

## Setup

```bash
# Activate the project venv (Python 3.11)
source .venv/bin/activate

# Install hard dependencies
pip install -r requirements.txt

# OpenAI-compatible API key (required for generation, not for retrieval-only evals)
export OPENAI_API_KEY=sk-...
# Optional: point at any OpenAI-compatible endpoint (Azure, OpenRouter,
# Together, Groq, LM Studio at http://localhost:1234/v1, Ollama at
# http://localhost:11434/v1, vLLM, etc.). Leave unset for OpenAI proper.
# export OPENAI_BASE_URL=https://openrouter.ai/api/v1
```

Optional extras (only needed for some configs):

```bash
pip install -r requirements-optional.txt   # voyage embeddings, gradio UI
export VOYAGE_API_KEY=...                  # only if running voyage_3_large
```

## Build the default index

```bash
python -m ingest --config experiments/default.yaml
```

Parses `docs/taxonomy_faqs_cleaned.md`, chunks per `### Question`, embeds with
`bge-small-en-v1.5`, builds a parallel BM25 index, persists to `index/default/`.
The `index/` directory is gitignored — rebuild from a fresh clone in seconds
(bge-small) or a few minutes (bge-large).

## Ask a question

```bash
python -m app "What does the EU Taxonomy require for Article 8 reporting?"
```

Switch configs:

```bash
python -m app --config experiments/embed_bge_large.yaml "..."
```

### Switching providers (Groq, OpenAI, OpenRouter, local LLMs)

The generator backend is provider-agnostic. Each experiment YAML can pin a
`generation.base_url` and `generation.model`; the same `OPENAI_API_KEY` env
var holds whichever provider's key is in play (the OpenAI SDK sends it as
`Authorization: Bearer …`, no matter the upstream).

`experiments/groq.yaml` ships as a worked example pointing at
`https://api.groq.com/openai/v1` with `llama-3.3-70b-versatile`. To use it:

```bash
export OPENAI_API_KEY=<your-groq-key>
python -m app --config experiments/groq.yaml "What does Article 8 require?"
```

To target other providers, copy `experiments/groq.yaml` and change the two
fields under `generation:`:

| Provider | `base_url` | sample `model` |
|---|---|---|
| OpenAI proper | (omit; default) | `gpt-4o-mini` |
| Azure OpenAI | `https://<resource>.openai.azure.com/openai/deployments/<dep>` | (deployment name) |
| OpenRouter | `https://openrouter.ai/api/v1` | `anthropic/claude-3.5-sonnet` |
| Together AI | `https://api.together.xyz/v1` | `meta-llama/Llama-3.3-70B-Instruct-Turbo` |
| LM Studio (local) | `http://localhost:1234/v1` | (whatever you loaded) |
| Ollama (local) | `http://localhost:11434/v1` | `llama3.3` |
| vLLM (local) | `http://localhost:8000/v1` | (served model id) |

If the top-1 retrieval similarity falls below the configured `tau_low`, the
pipeline short-circuits and returns the canonical refusal
(`I don't know based on the provided FAQs.`) without calling the LLM.

## Web UI (optional)

```bash
pip install -r requirements-optional.txt
.venv/bin/python ui.py
```

Opens a small Gradio surface at the URL it prints (usually
`http://127.0.0.1:7860`). If `OPENAI_API_KEY` is unset the UI degrades to
retrieval-only and shows the top-5 FAQ matches without calling the LLM.

## Run the experiment matrix

Per-config:

```bash
python -m evals.run --config experiments/<name>.yaml --retrieval-only
```

All configs at once:

```bash
python -m evals.run --all --retrieval-only
```

Results (per-case raw + aggregated summary) land in `evals/results/`;
[`evals/RESULTS.md`](evals/RESULTS.md) is regenerated from the merged set on
each run. Generation + LLM-as-judge metrics are gated on `OPENAI_API_KEY`
and described in [`DEFERRED.md`](DEFERRED.md).

## Results in 30 seconds

(See [`evals/RESULTS.md`](evals/RESULTS.md) for the full matrix; this is the
distilled story.)

### Retrieval

- **Best hit@5 = 0.972** (2-way tie between `rerank` and `max`). The reranker
  delivers +1.4 pp over the best non-reranker config — *below* the architecture's
  5 pp promotion threshold, so the reranker stays a variant, not the default.
- **Best MRR = 0.916** (`embed_bge_large`). bge-large is 9× slower per query
  than bge-small but lifts hit@1 from 0.833 to 0.875. Different headline →
  different winner.
- **Counter-prediction**: section-prepended chunking *hurt* hit@5 by 4.1 pp
  and adversarial by 12.5 pp. The architect's prediction of a small lift
  flipped sign on this corpus.
- **Stacking is sub-additive on hit@5**: `max` ties `rerank` exactly. The
  reranker absorbs all available retrieval headroom.
- **Fine-tune trigger does not fire** (hit@5 = 0.972 ≫ 0.85). Phase 5 is
  intentionally skipped per architecture §10.

### Generation (gpt-4o-mini through OpenAI proper, gpt-4o-mini also as LLM-judge)

- **Generation quality is essentially flat across configs**: faithfulness
  4.65–4.79 / 5, correctness 4.49–4.68. The retrieval matrix's hit@5 winners
  (`max`, `rerank`) edge out by 0.02–0.05 — within LLM-judge noise. The
  corpus is small and clean enough that any reasonable retrieval feeds
  gpt-4o-mini well.
- **`retrieval_hybrid` is the practical winner**: matches `max` (the absolute
  empirical winner) on faith/corr within noise, but runs **~5000× faster** per
  query (no CPU cross-encoder). For production, hybrid RRF + V1 prompt is the
  recommended config.
- **Prompt variants both lose to V1**: V2 (one-shot refusal example) ties
  V1 on faith but introduces context-length errors. V3 (verbatim citation
  instruction) actively hurts — faith down 0.12, correctness down 0.14, more
  FP refusals. The simple prompt wins.
- **Bigger embedder hurts generation despite better hit@1**: `embed_bge_large`
  loses 0.12 on faith and 0.12 on correctness vs `default` even though it
  retrieves better. Broader retrieval = more ambiguous context = lower
  faithfulness scores.

### Refusal

- **Model-side refusal does all the work**: all 7 generation configs achieve
  **perfect 8/8 OOS recall** through model self-refusal alone. Best refusal F1
  = **0.842** (4 configs tied), driven by 3 false-positive refusals on
  near-coverage in-scope cases (e.g. "small-company carve-out", nuclear-waste
  CDA-vs-CCDA disambiguation).
- **The configured `tau_low=0.30` floor never triggers**: bge-small returns
  top-1 cosine 0.47–0.78 even on truly off-topic queries. The architect's
  estimate (0.30–0.40 for bge-small) was too low for this corpus; layer 1 of
  the two-layer refusal architecture is currently dead weight, and layer 2
  (the model) is carrying 100% of the load.

## Hypotheses & assumptions

The architecture document (`docs/01_architecture.md`) commits to specific
predictions before any experiment runs. Holding them to the data:

| Hypothesis | Predicted | Measured | Verdict |
|---|---|---|---|
| Section-prepended chunking lifts hit@5 by 1–5 pp | + | -4.1 pp on hit@5, -12.5 pp on adversarial | **rejected** |
| BM25 ties dense on adversarial cases (exact-string signals) | yes | yes (0.750 = 0.750) | confirmed |
| Hybrid RRF matches or beats either component | yes | hit@5 ties dense at 0.958 | confirmed |
| Reranker lift on hit@5 is bounded by dense recall@20 | yes | +1.4 pp | confirmed |
| Reranker lift > 5 pp ⇒ promote to default | conditional | 1.4 pp < 5 pp | not promoted |
| Fine-tune triggers iff hit@5 < 0.85 ∧ gap < 2 pp | conditional | hit@5 = 0.972 | not triggered |
| Reranker lift on hit@5 ⇒ better generation faith/corr | + | rerank vs default: faith -0.05, corr -0.02 | **rejected** |
| bge-large lift on hit@1 ⇒ better generation faith/corr | + | embed_bge_large vs default: faith -0.12, corr -0.12 | **rejected** |
| V2 (one-shot refusal example) reduces FP refusals | + | flat (3 → 3 FPs); +2 context-length errors | rejected |
| V3 (verbatim citation) improves faithfulness | + | faith -0.12, refusal_F1 -0.08 | **rejected** |
| Stacking everything (max) helps generation | uncertain | yes, weakly: faith +0.02, corr +0.05 over default | weakly confirmed |
| Model-side refusal carries OOS recall ≥ 90 % | hopeful | 100 % (8/8 OOS) on all 7 generation configs | confirmed |
| `tau_low=0.30` floor catches off-topic queries | + | floor never triggers (OOS top-1 ∈ [0.47, 0.78]) | **rejected** |

Assumptions worth flagging to the reviewer:
- **Eval set size**. 80 cases is enough to rank configs but not enough to
  characterise the failure surface (especially the 8-case adversarial bucket
  where one case is 12.5 pp).
- **Single-shot scope**. No conversation history, no tool use, no agent loop.
  The architecture is intentionally a thin retrieve-then-generate, not a
  multi-step planner.
- **Eval set is Claude-paraphrased** from the corpus; numbers here are upper
  bounds versus production traffic where users phrase questions worse.
- **CPU-only constraint** on the host shaped some run-time decisions (the
  reranker took ~120 s/query); on a GPU the latency math reverses and the
  reranker becomes practical as the default. See `evals/RESULTS.md` §4.

## Future improvements

In approximate order of expected return per hour of effort:

1. **Recalibrate `tau_low`**. The configured floor (0.30) currently never
   triggers — on this corpus bge-small's top-1 cosine for OOS queries is
   0.47–0.78. The §8 floor calibration sweep in `evals/RESULTS.md` shows
   that raising to ~0.50 catches ~12% of OOS at the floor without an LLM
   call, modest but free. The deeper question is whether bge-small can
   discriminate OOS at all on this corpus; the answer for now is "barely",
   and layer 2 (model self-refusal) is doing the real work.
2. **Stronger judge model** (gpt-4o, claude-sonnet, etc.). Current judge is
   the same `gpt-4o-mini` as the generator → self-evaluation bias. Faith
   scores of 4.7+ are likely inflated by the bias. A separate, stronger
   judge would tighten the rankings — particularly on the 0.02–0.05 deltas
   between top configs that are currently within noise.
3. **GPU acceleration** for the reranker variants. Drops per-query latency
   from ~120 s on this 6-core CPU to ~50 ms on entry-level GPU; would make
   `rerank` and `max` viable as defaults rather than experimental variants.
4. **Investigate the 3 FP refusals** that recur across configs (`g_031`
   small-company carve-out; `a_001` nuclear waste; `a_005` iron-and-steel
   Article 8). All near-coverage corpus cases. The system reasonably plays
   safe; a "covered but indirect" prompt instruction might recover them
   without breaking the perfect 8/8 OOS recall.
5. **Fix `_format_context` truncation**. One generator error in Phase 4b
   was `context_length_exceeded` (case `m_005`, multi-FAQ pulling 5 long
   chunks → 176k tokens > gpt-4o-mini's 128k). Truncate per chunk before
   formatting.
6. **Grow the adversarial bucket** to ~16-24 cases stratified by trap type
   (near-duplicates, lexical-overlap, boundary-numeric, semantic-vs-surface).
   Current 8-case bucket is suggestive; doubling stabilises per-config
   differences.
7. **Voyage embedding variant** (`experiments/bge_large_hybrid_rerank.yaml`
   is already set up as a non-default-embedding template). Tests whether a
   hosted SOTA embedding meaningfully beats local bge-large.
8. **Real-user eval set**. Collect 50-100 real questions from
   sustainable-finance practitioners (typos, abbreviations, mixed languages)
   and re-score the matrix. The clean paraphrased eval set systematically
   over-estimates retrieval quality.
9. **Embedding fine-tune** (Phase 5) becomes interesting *only if* a real-user
   eval set drops hit@5 below the 0.85 trigger threshold. The fine-tune
   pipeline is sketched in `DEFERRED.md`.
10. **Chunk-level / semantic-cache layer**. The current build relies on
    the provider's automatic prefix caching, which only catches the system
    prompt because the retrieved context varies per query. For repeated
    near-duplicate queries (a real production pattern) a chunk-level cache
    or semantic-cache could pay off.

## Project layout

```
ingest.py           load + chunk + embed + BM25 → index/{name}/
retriever.py        dense / BM25 / hybrid (RRF) retrieval over a persisted index
reranker.py         optional cross-encoder reranker (noop default)
generator.py        OpenAI-compatible chat call (configurable base_url)
app.py              CLI entrypoint
ui.py               minimal Gradio web UI (optional)
prompts/            system prompts (v1 / v2 / v3 — git-tracked iterations)
experiments/        small YAML configs, one per matrix variant
evals/              cases.jsonl + run.py + metrics + RESULTS.md
docs/               architecture + corpus + (this README points at 01_architecture.md)
index/              built indexes (gitignored, rebuilt on demand)
DEFERRED.md         Phase 4b plan (API-gated work)
```

## Adding eval cases / configs

Eval set lives at `evals/cases.jsonl`. Each line is a JSON object with the
schema in [`evals/README.md`](evals/README.md): `id`, `category`, `question`,
`expected_chunk_ids`, `expected_answer_themes`, `expected_behaviour`, `notes`.
After adding a case, run `.venv/bin/python evals/validate_cases.py` to check
the schema and confirm `expected_chunk_ids` resolve.

The matrix is reproduced from `experiments/*.yaml` files. Copy `default.yaml`
and tweak — see `experiments/embed_bge_large.yaml` and `experiments/max.yaml`
for examples spanning the four axes (chunking / embedding / retrieval /
reranker).
