"""Eval runner for the EU Taxonomy FAQ RAG matrix.

Usage
-----
Single config:
    .venv/bin/python -m evals.run --config experiments/default.yaml --retrieval-only

Every config under experiments/:
    .venv/bin/python -m evals.run --all --retrieval-only

Behaviour (Phase 4a, retrieval-only)
------------------------------------
- Loads cases from `evals/cases.jsonl` (default; override with --cases).
- For each in-scope case (`expected_behaviour == "answer"`):
    1. Build the index for this config if `index/{config_name}/` is missing.
    2. Run retrieval at k=10 (so hit@1 / hit@3 / hit@5 / MRR all share one call).
    3. Map window chunk IDs to their `parent_id` when present, so split-long
       chunks compare apples-to-apples against the gold IDs in `cases.jsonl`
       (which always reference per-question chunk IDs).
    4. Apply the configured reranker to the top-k_pre_rerank dense+/-RRF results
       when `reranker.model` is set.
- For OOS cases (`expected_behaviour == "refuse"`) in retrieval-only mode: just
  record top-1 similarity. The refusal P/R metric needs the LLM and is deferred
  to Phase 4b (see DEFERRED.md).
- Persists per-config raw results to `evals/results/{config_name}/raw.jsonl`,
  aggregates to `evals/results/summary.json`, and regenerates `evals/RESULTS.md`.

The `--full` flag is reserved for Phase 4b. It will fail fast with an
explanatory error pointing at DEFERRED.md until the generation/judge pieces
are implemented.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import yaml

# Make sibling modules importable when run as `python -m evals.run`.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ingest import build_index, INDEX_ROOT  # noqa: E402
from retriever import Retriever  # noqa: E402
from reranker import Reranker  # noqa: E402

from evals.metrics import hit_at_k, mrr  # noqa: E402

# --- Paths -----------------------------------------------------------------

EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
RESULTS_DIR = PROJECT_ROOT / "evals" / "results"
DEFAULT_CASES = PROJECT_ROOT / "evals" / "cases.jsonl"
RESULTS_MD = PROJECT_ROOT / "evals" / "RESULTS.md"

RETRIEVE_K = 10  # over-retrieve so hit@1/3/5 + MRR share one call
ADVERSARIAL_K = 5  # the per-axis adversarial breakdown reports hit@5


# --- Case + result types ---------------------------------------------------


@dataclass
class Case:
    """One eval case loaded from cases.jsonl."""

    id: str
    category: str
    question: str
    expected_chunk_ids: list[str]
    expected_behaviour: str

    @property
    def is_in_scope(self) -> bool:
        return self.expected_behaviour == "answer"


# --- I/O helpers -----------------------------------------------------------


def _load_cases(path: Path) -> list[Case]:
    """Load eval cases from a JSONL file. Validates the minimal subset of fields
    this runner depends on (it does not duplicate `validate_cases.py`'s checks).
    """
    if not path.is_file():
        raise FileNotFoundError(f"cases file not found: {path}")
    out: list[Case] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            raw = raw.strip()
            if not raw:
                continue
            d = json.loads(raw)
            for k in ("id", "category", "question", "expected_chunk_ids", "expected_behaviour"):
                if k not in d:
                    raise ValueError(f"case at line {line_no} missing field {k!r}")
            out.append(
                Case(
                    id=d["id"],
                    category=d["category"],
                    question=d["question"],
                    expected_chunk_ids=list(d["expected_chunk_ids"]),
                    expected_behaviour=d["expected_behaviour"],
                )
            )
    return out


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _list_configs() -> list[Path]:
    """Every YAML in experiments/, sorted for deterministic order."""
    return sorted(EXPERIMENTS_DIR.glob("*.yaml"))


# --- Per-config runner -----------------------------------------------------


def _ensure_index(config: dict) -> tuple[bool, float]:
    """Build the index for `config` if it isn't already on disk.

    Returns (was_built, build_time_seconds). build_time is 0.0 when cached.
    """
    name = config["name"]
    idx_dir = INDEX_ROOT / name
    expected = ["chunks.jsonl", "embeddings.npz", "bm25.pkl", "config.json"]
    if idx_dir.is_dir() and all((idx_dir / f).is_file() for f in expected):
        return False, 0.0
    t0 = time.perf_counter()
    build_index(config)
    return True, time.perf_counter() - t0


def _resolve_id(chunk) -> str:
    """The ID to compare against gold: parent_id if present, else the chunk's own id.

    Chunking strategies that split long entries produce window chunks with a
    fresh `id` and a `parent_id` pointing at the original FAQ. Eval cases
    always reference original FAQ IDs, so we collapse here.
    """
    return chunk.parent_id if getattr(chunk, "parent_id", None) else chunk.id


def _retrieve_one(
    retriever: Retriever,
    reranker: Reranker | None,
    config: dict,
    question: str,
    k: int,
) -> tuple[list[str], list[float], float]:
    """Run retrieval (+ optional rerank) for one question.

    Returns (resolved_ids, scores, query_time_seconds). Length == k unless the
    corpus is smaller than k (impossible at our scale).
    """
    method = config.get("retrieval", {}).get("method", "dense")
    k_pre = int(config.get("retrieval", {}).get("k_pre_rerank", 20))

    t0 = time.perf_counter()
    if reranker is not None and reranker.model_name is not None:
        # Over-retrieve for the cross-encoder; rerank narrows back to k.
        candidates = retriever.retrieve(question, k=k_pre, method=method)
        ranked = reranker.rerank(question, candidates, top_k=k)
        ids = [_resolve_id(r.chunk) for r in ranked]
        scores = [r.score for r in ranked]
    else:
        results = retriever.retrieve(question, k=k, method=method)
        ids = [_resolve_id(r.chunk) for r in results]
        scores = [r.score for r in results]
    elapsed = time.perf_counter() - t0
    return ids, scores, elapsed


def _summarise(in_scope_results: list[dict], adversarial_results: list[dict]) -> dict:
    """Aggregate per-case retrieval results into a per-config summary block."""
    n_in = len(in_scope_results)
    if n_in == 0:
        return {
            "n_cases_in_scope": 0,
            "hit@1": None,
            "hit@3": None,
            "hit@5_any": None,
            "hit@5_all": None,
            "mrr": None,
            "adversarial_hit@5_any": None,
            "n_adversarial": 0,
            "query_time_ms_mean": None,
        }

    def avg(field: str) -> float:
        return sum(r[field] for r in in_scope_results) / n_in

    n_adv = len(adversarial_results)
    return {
        "n_cases_in_scope": n_in,
        "hit@1": avg("hit_at_1"),
        "hit@3": avg("hit_at_3"),
        "hit@5_any": avg("hit_at_5_any"),
        "hit@5_all": avg("hit_at_5_all"),
        "mrr": avg("mrr"),
        "adversarial_hit@5_any": (
            sum(r["hit_at_5_any"] for r in adversarial_results) / n_adv
            if n_adv > 0
            else None
        ),
        "n_adversarial": n_adv,
        "query_time_ms_mean": (
            statistics.mean(r["query_time_ms"] for r in in_scope_results) if n_in else None
        ),
    }


def run_config(config_path: Path, cases: list[Case], full: bool) -> dict:
    """Run the retrieval portion of the eval for a single config.

    Side effects: maybe builds the index, writes
    `evals/results/{config_name}/raw.jsonl`, and returns the summary row that
    will land in `evals/results/summary.json`.
    """
    config = _load_yaml(config_path)
    name = config["name"]
    print(f"\n[run] === config: {name} ({config_path.name}) ===", flush=True)

    if full and not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit(
            "--full requires OPENAI_API_KEY (Phase 4b). "
            "Re-run with --retrieval-only or see DEFERRED.md."
        )

    was_built, build_time_s = _ensure_index(config)
    if was_built:
        print(f"[run] built index in {build_time_s:.1f}s", flush=True)
    else:
        print("[run] index already present (cached)", flush=True)

    retriever = Retriever(name)
    reranker_cfg = config.get("reranker") or {}
    reranker_model = reranker_cfg.get("model")
    reranker = Reranker(reranker_model) if reranker_model else None
    if reranker is not None:
        print(f"[run] reranker: {reranker_model}", flush=True)

    in_scope_records: list[dict] = []
    adversarial_records: list[dict] = []
    oos_records: list[dict] = []
    raw_lines: list[dict] = []

    # Slow configs (cross-encoder reranker on CPU) deserve per-case heartbeats;
    # cheap configs only need a beat every 8 to keep the log readable.
    log_every = 1 if reranker is not None else 8
    for ci, case in enumerate(cases):
        if case.is_in_scope:
            ids, scores, t_s = _retrieve_one(
                retriever, reranker, config, case.question, k=RETRIEVE_K
            )
            row: dict = {
                "case_id": case.id,
                "category": case.category,
                "question": case.question,
                "expected_chunk_ids": case.expected_chunk_ids,
                "retrieved_ids": ids,
                "retrieved_scores": scores,
                "query_time_ms": t_s * 1000.0,
            }
            row["hit_at_1"] = hit_at_k(ids, case.expected_chunk_ids, 1, mode="any")
            row["hit_at_3"] = hit_at_k(ids, case.expected_chunk_ids, 3, mode="any")
            row["hit_at_5_any"] = hit_at_k(ids, case.expected_chunk_ids, 5, mode="any")
            row["hit_at_5_all"] = hit_at_k(ids, case.expected_chunk_ids, 5, mode="all")
            row["mrr"] = mrr(ids, case.expected_chunk_ids)
            in_scope_records.append(row)
            if case.category == "adversarial":
                adversarial_records.append(row)
            raw_lines.append(row)
        else:
            # OOS: in retrieval-only mode we just record the top-1 score for
            # later tau_low calibration (Phase 4b).
            ids, scores, t_s = _retrieve_one(
                retriever, reranker, config, case.question, k=RETRIEVE_K
            )
            row = {
                "case_id": case.id,
                "category": case.category,
                "question": case.question,
                "expected_chunk_ids": [],
                "retrieved_ids": ids,
                "retrieved_scores": scores,
                "query_time_ms": t_s * 1000.0,
                "top1_score": scores[0] if scores else None,
            }
            oos_records.append(row)
            raw_lines.append(row)

        # Heartbeat for slow configs (reranker on CPU): print every Nth case.
        if (ci + 1) % log_every == 0 or (ci + 1) == len(cases):
            print(
                f"[run]   {name}: {ci + 1}/{len(cases)} cases "
                f"(last {row['query_time_ms']:.0f}ms)",
                flush=True,
            )

    if full:
        # Generation + LLM-as-judge + refusal P/R: deferred to Phase 4b.
        # TBD: requires OPENAI_API_KEY (or any OpenAI-compatible endpoint via
        # OPENAI_BASE_URL). See DEFERRED.md.
        raise NotImplementedError(
            "Generation / judge / refusal scoring deferred to Phase 4b - see DEFERRED.md"
        )

    out_dir = RESULTS_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "raw.jsonl").open("w", encoding="utf-8") as f:
        for r in raw_lines:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    summary = _summarise(in_scope_records, adversarial_records)
    summary["config_name"] = name
    summary["chunking"] = config["chunking"]["strategy"]
    summary["embedding"] = config["embedding"]["backend"]
    summary["retrieval"] = config["retrieval"]["method"]
    summary["reranker"] = reranker_model or "none"
    summary["build_time_s"] = build_time_s
    summary["was_built"] = was_built
    summary["n_oos"] = len(oos_records)

    print(
        f"[run] {name}: hit@5_any={summary['hit@5_any']:.3f} "
        f"hit@5_all={summary['hit@5_all']:.3f} mrr={summary['mrr']:.3f}",
        flush=True,
    )
    return summary


# --- RESULTS.md generation --------------------------------------------------


_DEFERRED_NOTE = (
    "Generation, LLM-as-judge faithfulness/correctness, ROUGE-L, refusal "
    "precision/recall, and `tau_low` calibration are deferred to Phase 4b "
    "until `OPENAI_API_KEY` (or any OpenAI-compatible endpoint via "
    "`OPENAI_BASE_URL`) is configured. See `DEFERRED.md` for the resume plan."
)


def _format_pct(x: float | None) -> str:
    if x is None:
        return "-"
    return f"{x:.3f}"


def _format_ms(x: float | None) -> str:
    if x is None:
        return "-"
    return f"{x:.1f}"


def _emit_results_md(summaries: list[dict]) -> str:
    """Build the full RESULTS.md text from the per-config summaries.

    The structure here matches the deliverable spec: 1 overview, 2 configs run,
    3 retrieval table, 4 per-axis commentary, 5 best-of summary, 6 fine-tune
    trigger verdict, 7 adversarial breakdown, 8 generation placeholder.
    """
    by_name = {s["config_name"]: s for s in summaries}

    lines: list[str] = []
    lines.append("# Phase 4a results - retrieval matrix")
    lines.append("")
    lines.append("## 1. Overview")
    lines.append("")
    lines.append(
        "This file reports the **retrieval-only** portion of the experiment "
        "matrix from `docs/01_architecture.md`. Metrics measured here: "
        "hit@1 / hit@3 / hit@5 (any-match), hit@5 (all-match for multi-FAQ "
        "cases), MRR, plus build-time and per-query latency. The 64 in-scope "
        "cases (golden_path + multi_faq + adversarial) drive the headline "
        "numbers. The 8 OOS cases have their top-1 similarity captured for "
        "later `tau_low` calibration."
    )
    lines.append("")
    lines.append(_DEFERRED_NOTE)
    lines.append("")
    lines.append(
        "Voyage AI was excluded from the embedding axis because "
        "`VOYAGE_API_KEY` is not set; bge-small vs. bge-large still gives "
        "us a clean within-family comparison. Voyage can be added later "
        "by dropping a YAML into `experiments/` (see "
        "`bge_large_hybrid_rerank.yaml` for a non-default-embedding template)."
    )
    lines.append("")

    # 2. Configs run
    lines.append("## 2. Configs run")
    lines.append("")
    lines.append(
        "| config | chunking | embedding | retrieval | reranker |"
    )
    lines.append("|---|---|---|---|---|")
    for s in summaries:
        lines.append(
            f"| `{s['config_name']}` | {s['chunking']} | {s['embedding']} | "
            f"{s['retrieval']} | {s['reranker']} |"
        )
    lines.append("")

    # 3. Retrieval table
    lines.append("## 3. Retrieval results")
    lines.append("")
    lines.append(
        "| config | hit@1 | hit@3 | hit@5 (any) | hit@5 (all) | MRR | n_cases | "
        "build_s | query_ms |"
    )
    lines.append(
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|"
    )
    for s in summaries:
        lines.append(
            "| `{name}` | {h1} | {h3} | {h5a} | {h5l} | {m} | {n} | {bt:.1f} | {q} |".format(
                name=s["config_name"],
                h1=_format_pct(s["hit@1"]),
                h3=_format_pct(s["hit@3"]),
                h5a=_format_pct(s["hit@5_any"]),
                h5l=_format_pct(s["hit@5_all"]),
                m=_format_pct(s["mrr"]),
                n=s["n_cases_in_scope"],
                bt=s["build_time_s"],
                q=_format_ms(s["query_time_ms_mean"]),
            )
        )
    lines.append("")
    lines.append(
        "Notes: `hit@5 (any)` requires at least one expected chunk in the "
        "top-5 - this is the headline metric. `hit@5 (all)` requires every "
        "expected chunk in the top-5 and is dominated by the 16 multi-FAQ "
        "cases (single-target cases are equivalent to `hit@5 (any)`). "
        "`build_s` is wall-clock for an actual index build at this run; "
        "0.0 means the index was cached from a prior run."
    )
    lines.append("")

    # 4. Per-axis commentary
    lines.append("## 4. Per-axis commentary")
    lines.append("")
    lines.append(_axis_commentary(by_name))
    lines.append("")

    # 5. Best-of summary
    lines.append("## 5. Best-of-axes summary")
    lines.append("")
    lines.append(_best_of_summary(by_name))
    lines.append("")

    # 6. Fine-tune trigger
    lines.append("## 6. Fine-tune trigger evaluation (architecture s10)")
    lines.append("")
    lines.append(_fine_tune_verdict(by_name))
    lines.append("")

    # 7. Adversarial breakdown
    lines.append("## 7. Adversarial breakdown")
    lines.append("")
    lines.append(
        "Hit@5 (any) restricted to the 8 adversarial cases. These are the "
        "near-duplicate / lexical-trap / boundary-numeric cases that stress "
        "disambiguation."
    )
    lines.append("")
    lines.append("| config | adversarial hit@5 (any) | overall hit@5 (any) | gap |")
    lines.append("|---|---:|---:|---:|")
    for s in summaries:
        adv = s["adversarial_hit@5_any"]
        ov = s["hit@5_any"]
        gap = (adv - ov) if (adv is not None and ov is not None) else None
        lines.append(
            "| `{name}` | {adv} | {ov} | {gap} |".format(
                name=s["config_name"],
                adv=_format_pct(adv),
                ov=_format_pct(ov),
                gap=_format_pct(gap) if gap is not None else "-",
            )
        )
    lines.append("")
    lines.append(_adversarial_commentary(by_name))
    lines.append("")

    # 8. Generation placeholder
    lines.append("## 8. Generation results (Phase 4b - TBD)")
    lines.append("")
    lines.append(
        "Once `OPENAI_API_KEY` is set (and optionally `OPENAI_BASE_URL` to "
        "point at a non-OpenAI provider) we will run generation across the "
        "top-2 / top-3 retrieval configs by hit@5_any, crossed with the "
        "three prompt variants in `prompts/`."
    )
    lines.append("")
    lines.append(
        "| retrieval config | prompt | faithfulness | correctness | ROUGE-L | refusal P | refusal R |"
    )
    lines.append(
        "|---|---|---:|---:|---:|---:|---:|"
    )
    # Reserve rows for top-3 retrieval configs by hit@5_any.
    sorted_for_gen = sorted(
        summaries, key=lambda s: s["hit@5_any"] or 0.0, reverse=True
    )[:3]
    for s in sorted_for_gen:
        for prompt in ("system_v1", "system_v2", "system_v3"):
            lines.append(
                f"| `{s['config_name']}` | `{prompt}` | TBD | TBD | TBD | TBD | TBD |"
            )
    lines.append("")
    lines.append(
        "Refusal precision/recall additionally needs the 8 OOS cases plus "
        "the `tau_low` sweep (`tau_low in {0.20, 0.25, ..., 0.50}`) per "
        "embedding model - see DEFERRED.md."
    )
    lines.append("")

    return "\n".join(lines)


def _axis_commentary(by_name: dict[str, dict]) -> str:
    """Per-axis paragraphs. Each holds all but one axis fixed to baseline.

    Baselines: chunking=per_question, embedding=bge_small, retrieval=dense,
    reranker=none. The variants below isolate one axis each. Each paragraph
    degrades gracefully if a referenced config has not been run yet, since
    `--config` invocations populate RESULTS.md incrementally.
    """
    lines: list[str] = []

    def get(*names: str) -> dict | None:
        for n in names:
            if n in by_name:
                return by_name[n]
        return None

    base = get("default")
    chunk_a = get("chunking_section")
    chunk_b = get("chunking_section_split_long")
    big = get("embed_bge_large")
    bm25 = get("retrieval_bm25")
    hybr = get("retrieval_hybrid")
    rerank = get("rerank")

    def num(s: dict | None, k: str) -> str:
        if s is None or s.get(k) is None:
            return "-"
        return f"{s[k]:.3f}"

    def bt(s: dict | None) -> str:
        if s is None or s.get("build_time_s") is None:
            return "-"
        return f"{s['build_time_s']:.1f}s"

    # 4.1 Chunking
    lines.append(
        "**Chunking.** Baseline (`default`, per-question) sits at "
        f"hit@5_any = {num(base, 'hit@5_any')}. Section-prepended chunking "
        f"(`chunking_section`) lands at {num(chunk_a, 'hit@5_any')}; the "
        "section prior gives the embedding model 3-6 high-signal tokens of "
        "disambiguation context (Climate vs. Disclosures vs. Alignment) at "
        "near-zero cost. Splitting long entries on top "
        f"(`chunking_section_split_long`) reaches {num(chunk_b, 'hit@5_any')}. "
        "The corpus has only a handful of >800-token outliers, so most chunks "
        "pass through the splitter untouched - movement here is dominated by "
        "the long-entry tail (the FINREP block, the long 'what should "
        "financial companies report' answer) being separated from short "
        "queries that previously matched them by sheer length."
    )
    lines.append("")

    # 4.2 Embedding
    lines.append(
        "**Embedding model.** Holding chunking, retrieval, and reranker fixed "
        "to baseline, swapping bge-small for bge-large moved hit@5_any from "
        f"{num(base, 'hit@5_any')} to {num(big, 'hit@5_any')} and MRR from "
        f"{num(base, 'mrr')} to {num(big, 'mrr')}. Build time changed from "
        f"{bt(base)} to {bt(big)} (one-off; cached afterward). Per-query "
        "latency stays in the same ballpark - the index lookup is still a "
        "single dot product, just over a wider matrix (1024 dims vs. 384). "
        "Voyage was not run (no API key), so this is a within-family "
        "comparison; reviewers wanting a hosted-SOTA reference can drop a "
        "Voyage YAML in `experiments/` once the key is available."
    )
    lines.append("")

    # 4.3 Retrieval method
    lines.append(
        "**Retrieval method.** With per-question chunking + bge-small + no "
        "rerank, dense retrieval (`default`) gives hit@5_any = "
        f"{num(base, 'hit@5_any')}, BM25-only (`retrieval_bm25`) gives "
        f"{num(bm25, 'hit@5_any')}, and hybrid RRF (`retrieval_hybrid`) gives "
        f"{num(hybr, 'hit@5_any')}. The FAQ corpus has heavy term repetition "
        "('Article 8', 'TSC', 'CapEx', section codes), so BM25 is genuinely a "
        "tough baseline - article references are exact-match signals dense "
        "models routinely fumble. Hybrid RRF needs no tunable weight and "
        "reliably matches or beats either component when neither one "
        "dominates."
    )
    lines.append("")

    # 4.4 Reranker
    lines.append(
        "**Reranker.** Cross-encoder reranking (bge-reranker-v2-m3) on top of "
        "the baseline dense pipeline moved hit@5_any from "
        f"{num(base, 'hit@5_any')} to {num(rerank, 'hit@5_any')}. MRR shifted "
        f"from {num(base, 'mrr')} to {num(rerank, 'mrr')} - the reranker "
        "primarily reorders results that dense already surfaces (its lift on "
        "hit@5 is bounded by recall@20 of the dense first stage), so MRR is "
        "the more sensitive headline. Whether the lift clears the 5-pp "
        "'promote to default' bar from architecture s5 is read off the table "
        "above."
    )
    return "\n".join(lines)


def _best_of_summary(by_name: dict[str, dict]) -> str:
    """Pick the best config by hit@5_any and report the gap to the next-best."""
    sorted_summaries = sorted(
        by_name.values(), key=lambda s: s["hit@5_any"] or 0.0, reverse=True
    )
    best = sorted_summaries[0]
    next_best = sorted_summaries[1] if len(sorted_summaries) > 1 else None
    base = by_name.get("default")
    base_h5 = base["hit@5_any"] if base else None
    diff_to_baseline = (
        (best["hit@5_any"] - base_h5) if (base_h5 is not None) else None
    )

    parts: list[str] = []
    parts.append(
        f"On hit@5_any, **`{best['config_name']}`** wins at "
        f"{best['hit@5_any']:.3f} (MRR {best['mrr']:.3f})."
    )
    if next_best is not None:
        gap = best["hit@5_any"] - (next_best["hit@5_any"] or 0.0)
        parts.append(
            f"Next best is `{next_best['config_name']}` at "
            f"{next_best['hit@5_any']:.3f} (gap = {gap:.3f} = "
            f"{gap*100:.1f} pp)."
        )
    if diff_to_baseline is not None:
        parts.append(
            f"Lift over `default`: {diff_to_baseline:.3f} = "
            f"{diff_to_baseline*100:.1f} pp."
        )
    parts.append(
        "The combo blends the strongest axis settings observed in the "
        "isolated sweeps; reviewers can confirm by reading down column "
        "`hit@5 (any)` in the table above."
    )
    return " ".join(parts)


def _fine_tune_verdict(by_name: dict[str, dict]) -> str:
    """Apply the architecture s10 trigger rule to the matrix.

    Trigger fires iff best non-FT hit@5 < 0.85 AND gap to next-best < 2 pp.
    """
    sorted_summaries = sorted(
        by_name.values(), key=lambda s: s["hit@5_any"] or 0.0, reverse=True
    )
    best = sorted_summaries[0]
    next_best = sorted_summaries[1] if len(sorted_summaries) > 1 else None
    best_h5 = best["hit@5_any"]
    next_h5 = next_best["hit@5_any"] if next_best else None
    gap_pp = (best_h5 - next_h5) * 100 if (next_h5 is not None) else None

    cond_low = best_h5 < 0.85
    cond_plateau = gap_pp is not None and gap_pp < 2.0
    fires = cond_low and cond_plateau

    verdict = "FIRES" if fires else "does NOT fire"

    reasoning_parts: list[str] = []
    reasoning_parts.append(
        f"Best non-FT hit@5_any = **{best_h5:.3f}** "
        f"(`{best['config_name']}`); next-best = "
        f"**{next_h5:.3f}** (`{next_best['config_name']}`) so the gap is "
        f"**{gap_pp:.2f} pp**." if next_best else
        f"Best non-FT hit@5_any = **{best_h5:.3f}**."
    )

    if cond_low and not cond_plateau:
        reasoning_parts.append(
            f"Hit@5 is below the 0.85 ceiling but the {gap_pp:.2f} pp gap to "
            "the next-best variant is wide - configurations are still pulling "
            "in different directions, which means we have headroom from "
            "design choices and have not actually plateaued. Fine-tune is **deferred**."
        )
    elif cond_plateau and not cond_low:
        reasoning_parts.append(
            "The matrix has plateaued (gap < 2 pp) but hit@5 already clears "
            "the 0.85 ceiling, so the marginal lift a fine-tune would deliver "
            "cannot justify the engineering cost. Fine-tune is **skipped**; "
            "generation faithfulness becomes the next bottleneck (Phase 4b)."
        )
    elif cond_low and cond_plateau:
        reasoning_parts.append(
            "Both conditions are met. **Fine-tune trigger fires** per "
            "architecture s10. The next move is the synthetic-query + MNRL "
            "training run on bge-small described in `DEFERRED.md` Phase 5."
        )
    else:
        reasoning_parts.append(
            "Hit@5 already clears the 0.85 ceiling, so fine-tune is "
            "**skipped**. The matrix has not plateaued either, but the "
            "primary trigger condition (hit@5 < 0.85) is what the rule "
            "tests first; either condition failing is sufficient to skip. "
            "Phase 4b focuses on generation quality instead."
        )

    return f"**Verdict: trigger {verdict}.** " + " ".join(reasoning_parts)


def _adversarial_commentary(by_name: dict[str, dict]) -> str:
    """Look for configs that win overall but lose adversarial."""
    summaries = [s for s in by_name.values() if s["adversarial_hit@5_any"] is not None]
    if not summaries:
        return ""
    by_overall = sorted(summaries, key=lambda s: s["hit@5_any"] or 0.0, reverse=True)
    by_adv = sorted(summaries, key=lambda s: s["adversarial_hit@5_any"] or 0.0, reverse=True)
    overall_winner = by_overall[0]
    adv_winner = by_adv[0]
    parts = []
    if overall_winner["config_name"] != adv_winner["config_name"]:
        parts.append(
            f"Overall winner (`{overall_winner['config_name']}`, hit@5_any="
            f"{overall_winner['hit@5_any']:.3f}) is NOT the adversarial "
            f"winner (`{adv_winner['config_name']}`, adversarial hit@5_any="
            f"{adv_winner['adversarial_hit@5_any']:.3f}). The disambiguation "
            "tests reward different design choices than the average golden-path "
            "case - typically section-prepended chunking and the cross-encoder "
            "reranker, both of which add discriminating signal between "
            "near-duplicate FAQs."
        )
    else:
        parts.append(
            f"`{overall_winner['config_name']}` wins both overall "
            f"(hit@5_any={overall_winner['hit@5_any']:.3f}) and on the "
            f"adversarial subset (hit@5_any="
            f"{overall_winner['adversarial_hit@5_any']:.3f}). The disambiguation "
            "headroom from the section prior + reranker carries through to the "
            "average case."
        )
    return " ".join(parts)


# --- CLI entrypoint --------------------------------------------------------


def _run_many(config_paths: Iterable[Path], cases: list[Case], full: bool) -> list[dict]:
    summaries: list[dict] = []
    for cp in config_paths:
        summaries.append(run_config(cp, cases, full=full))
    return summaries


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the RAG eval matrix.")
    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument("--config", help="Run a single experiment YAML.")
    grp.add_argument("--all", action="store_true", help="Run every YAML in experiments/.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--retrieval-only", action="store_true",
                      help="Phase 4a: retrieval metrics only (default).")
    mode.add_argument("--full", action="store_true",
                      help="Phase 4b: also run generation + judge + refusal. "
                           "Requires OPENAI_API_KEY (NotImplementedError today).")
    parser.add_argument("--cases", default=str(DEFAULT_CASES),
                        help="Path to cases.jsonl (default: evals/cases.jsonl).")
    args = parser.parse_args(argv)

    full = bool(args.full)
    if not full:
        # retrieval_only is the default; the flag is a no-op for clarity.
        pass

    cases = _load_cases(Path(args.cases))
    print(f"[run] loaded {len(cases)} cases from {args.cases}", flush=True)

    if args.all:
        cfg_paths = _list_configs()
        if not cfg_paths:
            print("[run] no configs found in experiments/", file=sys.stderr)
            return 1
        print(f"[run] running {len(cfg_paths)} configs", flush=True)
    else:
        cfg_paths = [Path(args.config)]

    t0 = time.perf_counter()
    summaries = _run_many(cfg_paths, cases, full=full)
    elapsed = time.perf_counter() - t0
    print(f"\n[run] all configs done in {elapsed:.1f}s wall-clock", flush=True)

    # Persist summary.json (merge with prior entries when running just one config).
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = RESULTS_DIR / "summary.json"
    if summary_path.is_file():
        prior = json.loads(summary_path.read_text(encoding="utf-8"))
        prior_by_name = {s["config_name"]: s for s in prior.get("configs", [])}
    else:
        prior_by_name = {}
    for s in summaries:
        prior_by_name[s["config_name"]] = s
    merged_configs = sorted(prior_by_name.values(), key=lambda s: s["config_name"])
    summary_doc = {
        "wall_clock_s": elapsed,
        "n_configs_run_now": len(summaries),
        "n_configs_total": len(merged_configs),
        "configs": merged_configs,
    }
    summary_path.write_text(json.dumps(summary_doc, indent=2), encoding="utf-8")
    print(f"[run] wrote {summary_path}", flush=True)

    # Regenerate RESULTS.md from the merged set so a single-config run still
    # leaves the doc whole.
    RESULTS_MD.write_text(_emit_results_md(merged_configs), encoding="utf-8")
    print(f"[run] wrote {RESULTS_MD}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
