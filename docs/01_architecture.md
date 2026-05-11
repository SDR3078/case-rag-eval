# 01 — RAG Architecture for the EU Taxonomy FAQ Chatbot

Design document for a RAG evaluation case study. Audience: reviewers reading the
codebase top-to-bottom. Scope: single-shot Q&A over `docs/taxonomy_faqs_cleaned.md`.

**Corpus shape (measured, not assumed):**
- 747,859 bytes, 3,017 lines, 7 top-level `##` sections, 329 `### Question` entries.
- Per-entry length: mean ~8 lines, median 5, p95 ~20, **one 152-line outlier** (FINREP reconciliation table). Most entries are 1–4 short paragraphs.
- High lexical overlap across entries: terms like "Article 8", "TSC", "DNSH", "CDA", "Climate Delegated Act", "CapEx/OpEx", "NACE" recur in hundreds of questions. Discriminating signal is in the *qualifier* (e.g. "for forestry", "for financial undertakings", "outside the EU"), not the head noun.
- Inter-section overlap is real: the Complementary CDA, Disclosures DA, and Alignment Reporting sections all discuss Article 8 disclosures from different angles.

These three facts — short atomic entries, heavy term repetition, and cross-section overlap — drive every choice below.

---

## 1. Chunking strategies

### Default: per-`### Question` block
- **Definition.** One chunk per `### Question` entry. Payload = the `### …?` line plus everything until the next `###` or `##`. Keep the question line in the chunk text — it usually paraphrases the answer's topic and gives the embedding model a free anchor.
- **Why default.** The corpus authors already did the chunking work. Each block is atomic, self-contained, has a known-good answer, and almost never references another entry without being self-explanatory. With 329 entries averaging ~600–800 tokens, this fits comfortably in any embedding context (most are <512 tokens) and any LLM context (top-5 ≈ 3–4k tokens).
- **Strength.** Maximum precision for the typical case ("one user question = one FAQ entry").
- **Weakness.** No section context. Entry 1322 ("When and how should the Taxonomy disclosures be verified by external reviewers?") is identical in surface form to a similar question that could plausibly live under "Climate Delegated Act"; the section header carries the disambiguating context. Also fails the 152-line FINREP entry: it will dominate retrieval if not split.

### Variant A: per-question + parent `## Section` header prepended (recommended secondary)
- **Definition.** Chunk text = `"# {section}\n\n{question}\n\n{answer}"`, e.g. `"# Disclosures Delegated Act - General\n\n### What should financial companies report…\n\n…"`. Vector store metadata carries `section`, `question`, `chunk_id` separately.
- **Why I expect this to win.** Embedding models reward topical priors. Adding the section name injects 3–6 tokens of high-signal context that disambiguates entries with otherwise generic surface form ("How does this delegated act interact with…"). Cost is ~1% extra tokens.
- **When it would hurt.** If a user query is *cross-cutting* (e.g. "How is the term 'turnover' used across all delegated acts?"), prepending one section name biases toward whichever section mentions it most — but the dense vector still has the answer body, so the harm is small.
- **Verdict.** Run as a variant; expect a small but positive lift on hit@5.

### Variant B: long-entry splitting (sliding window for outliers only)
- **Definition.** Apply default chunking; for any chunk > N tokens (N ≈ 800), split into overlapping windows of 600 tokens with 100-token overlap, all sharing `parent_question_id`. At retrieval, dedupe by `parent_question_id` (keep the highest-scoring window).
- **Why bother.** The 152-line FINREP entry, the 56-line "what should financial companies report" entry, and a handful of 30–50 line entries dominate by length under cosine similarity (more tokens → embedding averages more concepts → matches more queries spuriously). Splitting confines a window's vector to one sub-topic.
- **When it would hurt.** Splitting tiny entries (most of the corpus). Hence the conditional rule: **only split chunks > 800 tokens**, leaving 320+ entries untouched.
- **Verdict.** Cheap insurance against length bias. Worth running as a variant on top of Variant A.

### Variant C: semantic chunking (de-prioritised)
- LangChain-style: split on embedding-similarity discontinuities. Not justified here. The markdown's `###` boundaries already encode semantic units that the authors hand-curated. Semantic chunking would re-discover roughly the same boundaries at higher cost and lower stability across runs. **Skip unless retrieval ceiling forces it.**

**Decision.** Default = per-question. Run Variants A and B in the matrix. C is out of scope unless results stall.

---

## 2. Embedding models

Three models in the matrix, chosen to span the cost / quality / locality dimensions.

| Model | Dim | Host | Cost (~per 1M tokens) | Why include |
|---|---|---|---|---|
| **`bge-small-en-v1.5`** (default) | 384 | local sentence-transformers | $0 | Strong baseline on MTEB, tiny (~33M params), runs CPU-only in seconds for 329 chunks. Surfaces what a local-only deployment can do. |
| **`bge-large-en-v1.5`** | 1024 | local sentence-transformers | $0 | Same family, materially stronger on retrieval benchmarks. Lets us isolate "model size" as the variable while holding the family constant. ~335M params, runs on CPU in a minute for the full index, ~200ms per query. |
| **Voyage `voyage-3-large`** (or `voyage-3`) | 1024 | API | ~$0.18 / 1M tokens | API-class quality. Tests whether a hosted SOTA embedding meaningfully beats local large. The corpus is ~200k tokens to embed once, plus negligible per-query cost — the entire eval costs cents. |

Rejected:
- **`all-MiniLM-L6-v2`** — too weak vs. `bge-small`; would only show up as a "look at how bad we used to have it" baseline, not interesting.
- **OpenAI `text-embedding-3-large`** — defensible, but Voyage is closer to the Anthropic ecosystem and the cost/quality tradeoff is similar enough that running both is redundant. **Run only if Voyage shows surprises and we want a sanity check.**
- **`e5-large-v2`, `instructor-xl`** — keep as fallbacks if `bge-large` underperforms; instructor-xl needs a good prompt template ("Represent the EU regulation question for retrieval:") which adds a tunable.

**Note on E5/BGE asymmetry.** Both families distinguish "passage" vs. "query" representations (BGE: prepend `"Represent this sentence for searching relevant passages: "` to *queries* only; E5: `"query: …"` / `"passage: …"`). The `retriever.py` interface must encode this asymmetry per-backend or recall@k will silently degrade.

**Default pick: `bge-small-en-v1.5`** for fast iteration, with `bge-large` and `voyage-3-large` as the comparison axis.

---

## 3. Vector store

**Pick: numpy array + cosine similarity, in-memory, pickled to disk.**

- 329 vectors × 1024 dims × float32 = **~1.3 MB**. A FAISS index for this is overkill; a Chroma server is malpractice. A single `np.ndarray` with normalised rows and `scores = embeddings @ query_vec` is faster than FAISS-flat at this scale (no index overhead) and trivially auditable.
- The index lives in `index/{config_name}.npz` alongside a parallel `chunks.jsonl` (chunk text + metadata, line-aligned with the array). Loading takes <50 ms.
- **Swap at 100x scale (~33k vectors).** Move to FAISS-flat (`IndexFlatIP`) — still exact, still in-memory, still <50 MB. The retriever interface should expose `add(vectors, metadata)` and `search(query_vec, k)` so the swap is a single class. We do *not* need an ANN index until ~1M vectors or sub-millisecond targets.
- **Why not Chroma even now.** Adds a dependency, a process, a schema, and a persistence story we don't need. Reviewers should be able to read the retriever in one sitting.

---

## 4. Retrieval method matrix

Four cells. Decide up-front which to actually run, given a small corpus.

| Method | Run? | Rationale |
|---|---|---|
| **Dense-only** (default) | **Yes** — primary | The corpus is well-written prose; embeddings should excel. This is the headline number. |
| **BM25-only** (`rank_bm25`) | **Yes** — must have | Two reasons: (1) FAQ corpora often have surprisingly strong term overlap between question and answer, so BM25 is a tough baseline; (2) numbers, article references ("Article 8(6)"), and section codes ("Section 4.5") are exact-match signals that dense models routinely fumble. If BM25 ties dense, that's an interesting result. |
| **Hybrid (RRF, k=60)** | **Yes** | Reciprocal Rank Fusion needs no tunable weight, works out of the box, and reliably matches or beats either component on heterogeneous corpora. Default fusion. |
| **Hybrid (weighted convex)** | Skip unless RRF underwhelms | One extra hyperparameter (alpha) to tune with no eval signal that justifies it at this scale. |
| **Dense + cross-encoder rerank** | **Yes** — see §5 | Top-20 dense → rerank to top-5. |
| **Hybrid + reranker** | **Yes** — final stack | Best-case configuration; the "max" config in the cost section. |

**Opinion:** the corpus has heavy term repetition ("Article 8", "TSC") which makes BM25's IDF-weighting valuable but also means dense models will find more value in the *qualifiers*. I expect hybrid + reranker > dense + reranker > dense > hybrid > BM25, with a small absolute spread. The interesting science is the spread, not the winner.

---

## 5. Optional reranker

**Pick: `BAAI/bge-reranker-v2-m3`** as a tested variant (not the default).

- Runs locally, ~568M params, ~50–150ms per query for top-20 reranking on CPU, single-digit ms on GPU.
- More accurate than `bge-reranker-base` on long queries; multilingual (free of charge given a corpus that occasionally quotes French/German legal references).
- **Why a variant, not default.** Latency budget. With 329 chunks and a strong retriever, the marginal lift from a cross-encoder may be small. Better to ship the headline number from the cheaper config and report the reranker as a +X% improvement at +Y ms cost. Reviewers see the tradeoff explicitly.
- **Decision rule.** If the reranker delivers > +5 percentage-points on hit@5 over the best non-reranker config, promote it to default. Otherwise it stays a variant.

---

## 6. System prompt skeleton

The prompt is one file (`prompts/system_v1.md`); iterations live in git. Sketch only — full text in the prompt file:

1. **Role.** "You are an assistant that answers questions about the EU Taxonomy Navigator FAQs."
2. **Hard rules (invariants).**
   - Answer **only** from the `<context>` block. Do not draw on prior knowledge.
   - If the context does not contain the answer, reply exactly: `"I don't know based on the provided FAQs."`
   - Cite the source: at the end of the answer, list the section heading(s) and FAQ question(s) used, e.g. `Source: "Disclosures Delegated Act - General" / "What should financial companies report under this disclosures delegated act?"`.
   - Quote sparingly; paraphrase, but never contradict the source.
3. **Output shape.** Plain text answer (no markdown headers, no bullet ceremony unless the source uses them). 1–3 short paragraphs typical.
4. **Refusal triggers.** (a) Context does not contain the topic; (b) Context contradicts itself and we can't reconcile; (c) The user asks for opinion / advice / current legal status / anything beyond what the FAQs say.

Three prompt variants for the experimentation matrix:
- **V1**: rules + 0 examples (above).
- **V2**: V1 + a single one-shot demonstrating refusal on an out-of-scope question.
- **V3**: V1 + an explicit "cite the section heading verbatim" instruction.

---

## 7. OpenAI-compatible chat-completions call shape

The backend is **provider-agnostic** via the OpenAI SDK. `base_url` is configurable per-config (or via `OPENAI_BASE_URL`) so the same code targets OpenAI proper, Azure OpenAI, OpenRouter, Together, Groq, LM Studio, Ollama, vLLM, etc. Reviewers swap providers by setting one env var or one YAML field — no code change.

- **Model.** `gpt-4o-mini` (default). Cheap, widely available, fast, and supported by every OpenAI-compatible provider. For higher-fidelity LLM-as-judge runs, swap to `gpt-4o` or any stronger model your provider exposes by setting `generation.model` in the YAML.
- **Prompt-prefix caching.** No explicit `cache_control` parameter exists in the OpenAI Chat Completions API. Providers that support prefix caching (OpenAI proper at ≥1024-token prefixes; some others) handle it automatically. The system prompt sits at the start of every call so it is the natural cacheable prefix; the retrieved-context block lives in the user message and varies per query.
- **`max_tokens`**: 600. FAQ answers are short; this is a hard ceiling that catches runaway generations. (Some newer reasoning models prefer `max_completion_tokens`; we keep `max_tokens` for broad provider compatibility.)
- **`temperature`**: 0. Determinism is essential for eval reproducibility.
- **`messages` shape**: a `system` message with the prompt content, followed by a `user` message containing the `<context>...</context>` block plus the question.
- **Streaming.** Off in eval (we need the whole answer for grading); enabling streaming in CLI/UI is a small follow-up.

A side benefit of the OpenAI-compatible abstraction: the LLM-as-judge step reuses the same `Generator` class (different `model` and `prompt_path`) regardless of where the judge ultimately runs.

---

## 8. Refusal logic

**Two-layer design, evaluated as one.**

1. **Soft floor at retrieval.** Compute the max similarity score of the top-1 chunk. If it falls below `tau_low` (calibrated, expected ~0.30–0.40 for `bge-small`, model-specific), short-circuit and emit the canonical refusal *without* calling the LLM. This handles obvious off-topic queries cheaply ("what's the weather in Paris?").
2. **Model-side refusal.** For all other queries, pass top-k context to the LLM with the prompt invariants from §6. The model is the final arbiter — it can refuse even on high-similarity matches if the retrieved text doesn't actually answer the user's question (common when a query is in-domain but its specific angle isn't covered).

**Why both, not just (2).** Without the floor, the LLM sees nonsense context for off-topic queries and occasionally hallucinates a plausible-sounding refusal-rationale rather than the canonical refusal string. Floor enforces the canonical string.

**Why not just (1).** Threshold-only refusal can't catch cases where retrieval surfaces a related-but-inadequate FAQ — only the model can read the entry and decide it doesn't answer the question.

**Evaluation plan.**
- Eval set includes **15–20 out-of-scope cases** mixed with 50–80 in-scope cases (target total: 80–100).
- Out-of-scope subset spans: (a) clearly unrelated ("recipe for risotto"), (b) adjacent but uncovered ("how do I file my taxes in Belgium?"), (c) in-domain but uncovered angle ("are there penalties for missing Article 8 disclosures?").
- Metrics: **refusal precision** (of all model refusals, fraction that were genuinely OOS) and **refusal recall** (of all OOS questions, fraction the system correctly refused). Track both `tau_low`-only refusals and model-driven refusals separately to attribute behaviour.
- Calibrate `tau_low` per embedding model on a held-out slice of the OOS subset; this is one of the few hyperparameters that legitimately needs tuning.

---

## 9. Cost / latency expectations

Per-query, order-of-magnitude. Assume ~80-token user question, ~3,000 tokens of retrieved context (top-5 chunks), ~250-token answer.

| Config | Embed | Retrieve | Rerank | Generate (`gpt-4o-mini`) | Total latency | Cost / query |
|---|---|---|---|---|---|---|
| **Default** (`bge-small`, dense, no rerank) | ~10 ms (CPU) | <1 ms (numpy) | — | ~0.7–1.5 s | **~0.7–1.5 s** | ~$0.0006 (input ~$0.00045, output ~$0.00015) |
| **Max** (`voyage-3-large`, hybrid + reranker, `gpt-4o-mini`) | ~80–150 ms (API) | <2 ms (numpy + BM25) | ~80 ms (CPU reranker) | ~0.7–1.5 s | **~0.9–1.7 s** | ~$0.0007 (Voyage adds < $0.0001/query; rest unchanged) |

Index build cost (one-off): ~200k tokens × $0.18/1M = **~$0.04 with Voyage**, $0 local. Negligible.

Eval cost: ≈$0.10 per `--full` config = **<$1 total** for the full matrix at `gpt-4o-mini` rates. Pricier providers (e.g. `gpt-4o`, `claude-3.5-sonnet` via OpenRouter) scale linearly; budget accordingly.

Latency dominator is the LLM call, not retrieval. Worth knowing — retrieval optimisation has a tiny ceiling.

---

## 10. Fine-tuning decision rule (CRITICAL)

**Trigger:** Fine-tune `bge-small-en-v1.5` on synthetic question→FAQ pairs **if and only if**:

> **The best-performing non-fine-tuned variant** (any combination of chunking + embedding + retrieval + reranker) achieves **hit@5 < 0.85** on the **in-scope eval subset** (~50–80 cases) **AND** the gap to the next-best variant is < 2 percentage points (i.e. we've plateaued, not just chosen a bad config).

If hit@5 ≥ 0.85, fine-tune is **skipped** — the marginal lift cannot justify the engineering cost, and generation quality (faithfulness, refusal) becomes the bottleneck instead.

**If triggered:**
- Generate ~1,000 synthetic queries by prompting the configured LLM with each FAQ chunk: "produce 3 paraphrased questions a user might ask, only answerable from this entry."
- Train with MNRL (multiple-negatives ranking loss) on `bge-small`, batch size 32, ~3 epochs, LR 2e-5. Hold out 10% of FAQs entirely (their synthetic queries become a true test set, not seen during training).
- Re-evaluate. **Promote** if hit@5 lifts ≥ 5 percentage points on the held-out FAQs (not the seen ones — that would be leakage).

**Concrete failure mode this rule prevents.** Without an explicit numeric trigger, "should we fine-tune?" becomes a vibes call at 11pm and gets skipped. With this rule, the eval pipeline answers the question.

---

## 11. Query flow (online path)

1. **User question received** via CLI or Gradio UI; trimmed, length-capped at 500 chars.
2. **Embed query** with the configured embedding model. Apply the model's query prefix (`"Represent this sentence for searching relevant passages: "` for BGE; `"query: "` for E5). Result: 384- or 1024-dim normalised vector.
3. **Sparse score** (if hybrid): tokenise query, score against pre-built BM25 index over the same chunks.
4. **Retrieve top-k**, k=20 (over-retrieve to feed the reranker). Cosine similarity for dense, BM25 raw scores for sparse, RRF-fuse if hybrid.
5. **Refusal floor check.** If top-1 similarity < `tau_low`, return canonical refusal string and stop. No LLM call.
6. **Rerank** (if enabled) with `bge-reranker-v2-m3`: score `(query, chunk_text)` pairs, take top-5.
7. **Assemble context.** Format top-5 chunks as `<context><doc id="…" section="…">…text…</doc>…</context>`, preserving section heading and question line so the model can cite.
8. **Generate.** Call the configured OpenAI-compatible chat-completions endpoint (default `gpt-4o-mini`) with the system prompt + user message containing context + question. `max_tokens=600`, `temperature=0`. Prefix caching, where the provider supports it, is automatic.
9. **Return** answer text + cited chunk IDs (for UI display and eval logging).

---

## 12. Diagram

```
                           OFFLINE INDEXING
                           ─────────────────
                          (run once per config)

  ┌──────────────────────────┐
  │ taxonomy_faqs_cleaned.md │ (3017 lines, 7 sections, 329 entries)
  └────────────┬─────────────┘
               │
               ▼
  ┌──────────────────────────┐    chunking strategy:
  │     ingest.py / chunk    │    - default: per-Q
  └────────────┬─────────────┘    - variant A: per-Q + section header
               │                  - variant B: split outliers >800 tok
               │
               ├──────────────► chunks.jsonl  (text + metadata)
               │
               ▼
  ┌──────────────────────────┐    embedding model:
  │  embed (per chunk)       │    - bge-small (default)
  └────────────┬─────────────┘    - bge-large
               │                  - voyage-3-large
               ▼
  ┌──────────────────────────┐    BM25 index built in parallel
  │   index.npz  (numpy)     │    over the same chunks.jsonl
  └──────────────────────────┘



                            ONLINE QUERY
                            ────────────
                          (per user question)

   user question
       │
       ▼
  ┌─────────┐    ┌──────────────────┐     ┌─────────────────┐
  │  embed  │───►│  dense top-20    │     │   BM25 top-20   │
  │  query  │    │ (numpy cosine)   │     │ (rank_bm25)     │
  └─────────┘    └────────┬─────────┘     └────────┬────────┘
                          │                        │
                          └────────┬───────────────┘
                                   ▼
                          ┌──────────────────┐
                          │   RRF fusion     │  (if hybrid; else passthrough)
                          └────────┬─────────┘
                                   │
                                   ▼
                  ┌────────────────────────────────┐
                  │ refusal floor: max_sim < tau ? │──► YES ──► canonical refusal
                  └───────────────┬────────────────┘            (no LLM call)
                                  │ NO
                                  ▼
                          ┌──────────────────┐
                          │  rerank to top-5 │  (optional, bge-reranker-v2-m3)
                          └────────┬─────────┘
                                   ▼
                          ┌──────────────────┐
                          │  assemble        │
                          │  <context>…      │
                          └────────┬─────────┘
                                   ▼
              ┌────────────────────────────────────────┐
              │ OpenAI-compatible chat completion       │
              │  default: gpt-4o-mini  (base_url=any)   │
              │  - system prompt (cacheable prefix)     │
              │  - <context> + question (user message)  │
              │  - max_tokens=600, temperature=0        │
              └────────────────────┬───────────────────┘
                                   ▼
                          ┌──────────────────┐
                          │  answer + cites  │
                          └──────────────────┘
```

---

## 13. Open questions

Decisions that need experiment data, not more thinking. Listed with the data that would resolve each.

1. **Does Variant A (section-prepended chunks) actually beat the default?**
   *Resolved by:* hit@5 and MRR on the golden-path subset, default vs. Variant A, holding embedding model and retrieval method fixed. Expected: +1 to +5 pp lift; if 0, drop the variant.
2. **Is the reranker worth its latency?**
   *Resolved by:* hit@5 of `dense + reranker` vs. `dense alone`, both at top-5. If lift < 5 pp, reranker stays a variant; if ≥ 5 pp, promote to default.
3. **Does Voyage justify the API dependency over `bge-large`?**
   *Resolved by:* hit@5 and ROUGE-L of generated answer with Voyage embeddings vs. `bge-large`. If gap < 2 pp, default to `bge-large` (no API key needed for reviewers to run it).
4. **What is the right `tau_low` floor?**
   *Resolved by:* sweeping `tau_low ∈ {0.20, 0.25, …, 0.50}` per embedding model, plotting refusal precision/recall on the OOS subset. Pick the knee.
5. **Does fine-tuning trigger?**
   *Resolved by:* the rule in §10, run after the full matrix lands. Expectation: probably no — the corpus is too clean and `bge-large` should already saturate.
6. **Does prompt V2 (one-shot refusal) reduce hallucinated refusals on adjacent-but-uncovered queries?**
   *Resolved by:* refusal precision delta on the (b) and (c) subsets of the OOS eval. If V2 < V1, drop the example; if V2 > V1 by ≥ 3 pp, V2 becomes default.
7. **Does the long-entry split (Variant B) help or hurt overall?**
   *Resolved by:* hit@5 on a targeted subset of queries known to map to the FINREP outlier and the other long entries. If it lifts those without hurting the rest, ship it.
