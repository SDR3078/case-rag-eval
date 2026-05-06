"""Eval runner for the EU Taxonomy FAQ RAG matrix.

Usage
-----
Single config (retrieval only):
    .venv/bin/python -m evals.run --config experiments/default.yaml --retrieval-only

Single config with generation, judge, ROUGE-L, refusal P/R:
    OPENAI_API_KEY=... .venv/bin/python -m evals.run --config experiments/groq.yaml --full

Every config under experiments/:
    .venv/bin/python -m evals.run --all --retrieval-only

Behaviour
---------
For each in-scope case (`expected_behaviour == "answer"`):
    1. Build the index for this config if `index/{config_name}/` is missing.
    2. Run retrieval at k=10 (so hit@1 / hit@3 / hit@5 / MRR share one call).
    3. Map window chunk IDs to their `parent_id` when present, so split-long
       chunks compare apples-to-apples against the gold IDs in `cases.jsonl`
       (which always reference per-question chunk IDs).
    4. Apply the configured reranker when `reranker.model` is set.
    5. (--full only) Generate an answer through the OpenAI-compatible generator,
       run the LLM-as-judge for faithfulness + correctness, compute ROUGE-L
       against the canonical FAQ chunk text(s), record refusal flags.

For OOS cases (`expected_behaviour == "refuse"`):
    - Always record top-1 similarity for `tau_low` calibration.
    - With --full, also generate (no judge) and record whether the system
      refused so refusal precision/recall and the tau_low sweep can land in
      §5 of RESULTS.md.

Per-config raw results go to `evals/results/{config_name}/raw.jsonl`;
aggregates merge into `evals/results/summary.json`; `evals/RESULTS.md` is
regenerated from the merged set on every run.
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
from generator import CANONICAL_REFUSAL, Generator  # noqa: E402

from evals.metrics import hit_at_k, mrr  # noqa: E402
from evals.judge import Judge  # noqa: E402
from evals.rouge import rouge_l_against_chunks  # noqa: E402

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
    expected_answer_themes: list[str]
    expected_behaviour: str

    @property
    def is_in_scope(self) -> bool:
        return self.expected_behaviour == "answer"


# `tau_low` values used to compute the floor-only refusal P/R sweep when --full.
# Architecture §8 calibration grid.
_TAU_LOW_SWEEP = (0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50)


def _is_refusal(answer: str) -> bool:
    """Match the canonical refusal substring case-insensitively.

    The system prompt instructs the model to reply *exactly* with the canonical
    string when the context doesn't cover the question, so substring match is
    the right granularity (some models append "Source: ..." after).
    """
    return CANONICAL_REFUSAL.lower() in (answer or "").lower()


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
                    expected_answer_themes=list(d.get("expected_answer_themes", [])),
                    expected_behaviour=d["expected_behaviour"],
                )
            )
    return out


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _list_configs() -> list[Path]:
    """Every YAML in experiments/, sorted, deduped by `name`.

    Multiple YAMLs may share a `name` (e.g. `groq.yaml` reuses `name: default`
    to share the index dir; only its `generation:` block differs). For `--all`
    we keep just the lexicographically first such YAML, so a matrix run
    doesn't silently overwrite earlier rows in `summary.json`. Use `--config`
    to target a specific YAML by path when names collide.
    """
    paths = sorted(EXPERIMENTS_DIR.glob("*.yaml"))
    seen: set[str] = set()
    deduped: list[Path] = []
    for p in paths:
        with p.open("r", encoding="utf-8") as f:
            name = (yaml.safe_load(f) or {}).get("name")
        if name in seen:
            print(
                f"[run] note: skipping {p.name} (name={name!r} already taken "
                f"by an earlier YAML); use --config to target it directly.",
                flush=True,
            )
            continue
        seen.add(name)
        deduped.append(p)
    return deduped


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


def _generate_and_judge(
    case: Case,
    ids: list[str],
    scores: list[float],
    chunks_by_id: dict,
    generator: Generator,
    judge: Judge,
    method: str,
    tau_low: float,
) -> dict:
    """Per case: refusal-floor check, generate, optionally judge, ROUGE-L.

    Returns the dict of new fields to merge into the per-case raw row. The
    helper handles in-scope (judged) and OOS (not judged, only refusal-detected)
    uniformly so the calling loop stays small.
    """
    # Reconstruct the top-5 RetrievalResult list the generator/judge expect.
    # The case loop discarded the live RetrievalResult objects; reconstruct
    # from the cached `chunks_by_id` so the generator sees the same chunks
    # that retrieval recorded.
    from retriever import RetrievalResult

    results: list[RetrievalResult] = []
    for cid, score in zip(ids[:5], scores[:5]):
        chunk = chunks_by_id.get(cid)
        if chunk is None:
            continue
        results.append(RetrievalResult(chunk=chunk, score=float(score), method="full"))

    top1 = scores[0] if scores else 0.0
    # Refusal floor only applies to dense (cosine) scores. BM25 and RRF scores
    # aren't bounded the same way, so the floor is dense-only by design.
    floor_refused = (method == "dense" and top1 < tau_low)

    out: dict = {"floor_refused": floor_refused}

    if floor_refused:
        out["answer"] = CANONICAL_REFUSAL
        out["gen_usage"] = {}
    else:
        try:
            gen_out = generator.generate(case.question, results)
            out["answer"] = gen_out["answer"]
            out["gen_usage"] = gen_out["usage"]
        except Exception as e:
            out["answer"] = f"[generator error: {type(e).__name__}: {e}]"
            out["gen_usage"] = {}

    out["model_refused"] = _is_refusal(out["answer"])
    out["refused"] = floor_refused or out["model_refused"]

    if case.is_in_scope:
        # In-scope cases are judged on faithfulness + correctness. If the
        # system refused on an in-scope case, score 0/0 (refusal is the wrong
        # behaviour for these); skip the judge call to save tokens.
        if out["refused"]:
            out["faithfulness"] = 0
            out["correctness"] = 0
            out["judge_rationale"] = "system refused on in-scope case"
        elif out["answer"].startswith("[generator error:"):
            # Generator hit an API error; the answer is a stub message, not
            # something the judge can evaluate. Mark unscoreable rather than
            # spend a judge call on garbage.
            out["faithfulness"] = -1
            out["correctness"] = -1
            out["judge_rationale"] = "generator error — judge skipped"
        else:
            try:
                j = judge.score(
                    question=case.question,
                    retrieved=results,
                    answer=out["answer"],
                    expected_themes=case.expected_answer_themes,
                )
                out["faithfulness"] = j["faithfulness"]
                out["correctness"] = j["correctness"]
                out["judge_rationale"] = j["rationale"]
            except Exception as e:
                out["faithfulness"] = -1
                out["correctness"] = -1
                out["judge_rationale"] = f"judge error: {type(e).__name__}"

        # ROUGE-L against the canonical FAQ chunk text(s). For multi-FAQ cases
        # the references are concatenated. Refused answers and generator-error
        # stubs score 0 by fiat — measuring overlap against an error message
        # produces meaningless small values.
        gen_errored = out["answer"].startswith("[generator error:")
        if out["refused"] or gen_errored:
            out["rouge_l"] = 0.0
        else:
            ref_texts = [
                chunks_by_id[i].text
                for i in case.expected_chunk_ids
                if i in chunks_by_id
            ]
            out["rouge_l"] = (
                rouge_l_against_chunks(out["answer"], ref_texts) if ref_texts else 0.0
            )

    return out


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


def _summarise_full(
    in_scope_results: list[dict],
    oos_results: list[dict],
) -> dict:
    """Aggregate generation metrics from a `--full` run.

    Returns faithfulness/correctness/ROUGE-L means, refusal P/R/F1 over all
    cases, and a `tau_low` floor-only sweep for refusal calibration (P and R
    columns only — F1 collapses the trade-off the sweep is designed to expose).
    Cells are `None` when generation data is missing (e.g. retrieval-only run).
    """
    # Filter to records with judge data (in-scope cases that had generation done).
    judged = [
        r for r in in_scope_results
        if "faithfulness" in r and r["faithfulness"] >= 0
    ]
    n_judged = len(judged)
    rouge_records = [r for r in in_scope_results if "rouge_l" in r]
    n_rouge = len(rouge_records)

    # Refusal P/R: positive class = "should refuse" (OOS).
    # TP = OOS AND refused; FN = OOS AND not refused;
    # FP = in-scope AND refused; TN = in-scope AND not refused.
    tp = sum(1 for r in oos_results if r.get("refused"))
    fn = sum(1 for r in oos_results if r.get("refused") is False)
    fp = sum(1 for r in in_scope_results if r.get("refused"))
    tn = sum(1 for r in in_scope_results if r.get("refused") is False)
    refusal_n = tp + fn + fp + tn

    def _safediv(a: int, b: int) -> float | None:
        return (a / b) if b > 0 else None

    precision = _safediv(tp, tp + fp)
    recall = _safediv(tp, tp + fn)
    if precision is not None and recall is not None and (precision + recall) > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = None

    # tau_low sweep: floor-only refusals, computed analytically from logged
    # top-1 scores across both in-scope and OOS records. The sweep is dense-only
    # because BM25/RRF top-1 scores aren't on the same scale; for non-dense
    # configs the sweep table simply collapses to "floor never triggers".
    sweep: list[dict] = []
    in_scope_top1 = [
        (r["retrieved_scores"][0] if r.get("retrieved_scores") else 0.0)
        for r in in_scope_results
    ]
    oos_top1 = [
        (r["retrieved_scores"][0] if r.get("retrieved_scores") else 0.0)
        for r in oos_results
    ]
    for tau in _TAU_LOW_SWEEP:
        floor_tp = sum(1 for s in oos_top1 if s < tau)
        floor_fn = len(oos_top1) - floor_tp
        floor_fp = sum(1 for s in in_scope_top1 if s < tau)
        sweep.append({
            "tau_low": tau,
            "floor_tp": floor_tp,
            "floor_fn": floor_fn,
            "floor_fp": floor_fp,
            "floor_precision": _safediv(floor_tp, floor_tp + floor_fp),
            "floor_recall": _safediv(floor_tp, floor_tp + floor_fn),
        })

    return {
        "n_generated": len(in_scope_results) + len(oos_results),
        "n_judged": n_judged,
        "faithfulness_mean": (
            sum(r["faithfulness"] for r in judged) / n_judged if n_judged else None
        ),
        "correctness_mean": (
            sum(r["correctness"] for r in judged) / n_judged if n_judged else None
        ),
        "rouge_l_mean": (
            sum(r["rouge_l"] for r in rouge_records) / n_rouge if n_rouge else None
        ),
        "refusal_precision": precision,
        "refusal_recall": recall,
        "refusal_f1": f1,
        "refusal_tp": tp,
        "refusal_fp": fp,
        "refusal_fn": fn,
        "refusal_tn": tn,
        "refusal_n": refusal_n,
        "tau_low_sweep": sweep,
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
            "--full requires OPENAI_API_KEY (generation + judge). "
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

    # `--full` mode: build the generator + judge once, before the case loop.
    # Both eagerly check OPENAI_API_KEY in their constructors, so we fail fast
    # if the env is misconfigured. Judge inherits model/base_url from the
    # generation block when not separately specified — this is biased but
    # cheap; see prompts/judge_v1.md and DEFERRED.md.
    generator: Generator | None = None
    judge: Judge | None = None
    chunks_by_id: dict | None = None
    method = config.get("retrieval", {}).get("method", "dense")
    tau_low = float(config.get("retrieval", {}).get("tau_low", 0.30))
    if full:
        gen_cfg = config["generation"]
        judge_cfg = config.get("judge") or {}
        generator = Generator(
            model=gen_cfg["model"],
            prompt_path=config["prompt"]["path"],
            max_tokens=int(gen_cfg.get("max_tokens", 600)),
            temperature=float(gen_cfg.get("temperature", 0.0)),
            base_url=gen_cfg.get("base_url"),
        )
        judge = Judge(
            model=judge_cfg.get("model") or gen_cfg["model"],
            prompt_path=judge_cfg.get("prompt_path", "prompts/judge_v1.md"),
            max_tokens=int(judge_cfg.get("max_tokens", 200)),
            temperature=float(judge_cfg.get("temperature", 0.0)),
            base_url=judge_cfg.get("base_url") or gen_cfg.get("base_url"),
        )
        chunks_by_id = retriever._chunks_by_id
        print(
            f"[run] full mode: generator={generator.model} "
            f"judge={judge.model} (base_url={generator.base_url or 'default'})",
            flush=True,
        )

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
            # later tau_low calibration. When `full`, we also run generation
            # so we can detect model-side refusals.
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

        # Generation + judge step (in-scope and OOS both run generation; only
        # in-scope cases hit the judge). Side effect: enriches `row` in place.
        if full and generator is not None and judge is not None and chunks_by_id is not None:
            gen_fields = _generate_and_judge(
                case=case,
                ids=ids,
                scores=scores,
                chunks_by_id=chunks_by_id,
                generator=generator,
                judge=judge,
                method=method,
                tau_low=tau_low,
            )
            row.update(gen_fields)

        # Heartbeat for slow configs (reranker on CPU): print every Nth case.
        if (ci + 1) % log_every == 0 or (ci + 1) == len(cases):
            extra = ""
            if full and "faithfulness" in row:
                extra = (
                    f" gen={row.get('faithfulness', '?')}/"
                    f"{row.get('correctness', '?')}"
                )
            elif full and case.is_in_scope is False and "refused" in row:
                extra = f" refused={row['refused']}"
            print(
                f"[run]   {name}: {ci + 1}/{len(cases)} cases "
                f"(last {row['query_time_ms']:.0f}ms){extra}",
                flush=True,
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

    def _fmt(x: float | None, spec: str = ".3f") -> str:
        return f"{x:{spec}}" if x is not None else "n/a"

    if full:
        gen_summary = _summarise_full(in_scope_records, oos_records)
        summary["generation"] = gen_summary
        summary["prompt_path"] = config["prompt"]["path"]
        summary["generator_model"] = (generator.model if generator else None)
        summary["judge_model"] = (judge.model if judge else None)
        print(
            f"[run] {name}: hit@5_any={_fmt(summary['hit@5_any'])} "
            f"hit@5_all={_fmt(summary['hit@5_all'])} mrr={_fmt(summary['mrr'])} "
            f"faith={_fmt(gen_summary['faithfulness_mean'], '.2f')} "
            f"corr={_fmt(gen_summary['correctness_mean'], '.2f')} "
            f"rouge_l={_fmt(gen_summary['rouge_l_mean'])} "
            f"refusal_F1={_fmt(gen_summary['refusal_f1'])} "
            f"(judged {gen_summary['n_judged']}/{summary['n_cases_in_scope']}, "
            f"refused TP={gen_summary['refusal_tp']}/{gen_summary['refusal_tp']+gen_summary['refusal_fn']})",
            flush=True,
        )
    else:
        print(
            f"[run] {name}: hit@5_any={_fmt(summary['hit@5_any'])} "
            f"hit@5_all={_fmt(summary['hit@5_all'])} mrr={_fmt(summary['mrr'])}",
            flush=True,
        )
    return summary


# --- RESULTS.md generation --------------------------------------------------


_DEFERRED_NOTE = (
    "Generation, LLM-as-judge faithfulness/correctness, ROUGE-L, refusal "
    "precision/recall, and `tau_low` calibration are populated once "
    "`OPENAI_API_KEY` (or any OpenAI-compatible endpoint via "
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

    Sections: 1 overview, 2 configs run, 3 retrieval table, 4 adversarial
    breakdown, 5 generation results. Comparative analysis (per-axis effects,
    best-of, fine-tune verdict) lives in `docs/01_architecture.md` §10
    rather than as auto-emitted prose.
    """
    has_generation = any(s.get("generation") for s in summaries)

    lines: list[str] = []
    lines.append("# Experiment matrix results")
    lines.append("")
    lines.append("## 1. Overview")
    lines.append("")
    lines.append(
        "This file reports the experiment matrix from `docs/01_architecture.md`. "
        "Retrieval metrics — hit@1 / hit@3 / hit@5 (any-match), hit@5 (all-match "
        "for multi-FAQ cases), MRR, build-time, per-query latency — are populated "
        "for every config in §3. Generation metrics — LLM-judge faithfulness + "
        "correctness, ROUGE-L baseline, refusal precision/recall, `tau_low` "
        "calibration — populate §5 only for configs run with `--full` "
        "(generation requires `OPENAI_API_KEY`)."
    )
    lines.append("")
    if not has_generation:
        lines.append(_DEFERRED_NOTE)
    else:
        lines.append(
            "**Generation status:** at least one config has been run with "
            "`--full`; per-config numbers appear in §5. Configs without "
            "generation data are listed with TBD."
        )
    lines.append("")
    lines.append(
        "Voyage AI was excluded from the embedding axis because "
        "`VOYAGE_API_KEY` is not set; bge-small vs. bge-large still gives "
        "us a clean within-family comparison. Voyage can be added later "
        "by dropping a YAML into `experiments/` (copy `max.yaml` and switch "
        "`embedding.backend` to `voyage_3_large`)."
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

    # 4. Adversarial breakdown
    lines.append("## 4. Adversarial breakdown")
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

    # 5. Generation results — populated when --full has been run on at least
    # one config. Configs without generation data show TBD rows so the reader
    # can see what's still to come.
    gen_configs = [s for s in summaries if s.get("generation")]
    lines.append("## 5. Generation results")
    lines.append("")
    if gen_configs:
        lines.append(
            "Per-config generation metrics (populated by `--full`). "
            "Faithfulness and correctness are 0–5 LLM-judge scores against the "
            "expected answer themes; ROUGE-L is a string-overlap baseline "
            "against the canonical FAQ chunk text(s). Refusal P/R is computed "
            "over the 80-case set (positive class = OOS that the system "
            "correctly refused). The judge defaults to the same model as the "
            "generator unless `judge.model` is set in the YAML — note the "
            "self-evaluation bias when both columns are filled by the same model."
        )
        lines.append("")
        lines.append(
            "| config | prompt | generator | judge | faithfulness | correctness | ROUGE-L | refusal P | refusal R | refusal F1 |"
        )
        lines.append(
            "|---|---|---|---|---:|---:|---:|---:|---:|---:|"
        )
        for s in sorted(gen_configs, key=lambda x: x["config_name"]):
            g = s["generation"]
            prompt = Path(s.get("prompt_path", "prompts/system_v1.md")).stem
            lines.append(
                "| `{cfg}` | `{prompt}` | `{gm}` | `{jm}` | {faith:.2f} | {corr:.2f} | {rouge:.3f} | {p} | {r} | {f1} |".format(
                    cfg=s["config_name"],
                    prompt=prompt,
                    gm=s.get("generator_model", "?"),
                    jm=s.get("judge_model", "?"),
                    faith=g["faithfulness_mean"] or 0.0,
                    corr=g["correctness_mean"] or 0.0,
                    rouge=g["rouge_l_mean"] or 0.0,
                    p=_format_pct(g["refusal_precision"]) if g["refusal_precision"] is not None else "-",
                    r=_format_pct(g["refusal_recall"]) if g["refusal_recall"] is not None else "-",
                    f1=_format_pct(g["refusal_f1"]) if g["refusal_f1"] is not None else "-",
                )
            )
        lines.append("")
        # Refusal calibration sub-table from the tau_low sweep. Pull the first
        # config's sweep — for non-dense retrieval configs, `tau_low` is dense-
        # only and the sweep is informative only on the dense rows.
        first_with_sweep = next(
            (s for s in gen_configs if s["generation"].get("tau_low_sweep")),
            None,
        )
        if first_with_sweep is not None:
            lines.append(
                f"### Refusal floor calibration — `{first_with_sweep['config_name']}`"
            )
            lines.append("")
            lines.append(
                "Floor-only refusal precision/recall at each `tau_low`, computed "
                "from logged top-1 dense scores. The configured value lives in "
                "the YAML; this sweep helps pick a calibrated knee."
            )
            lines.append("")
            lines.append("| tau_low | floor TP | floor FP | floor FN | precision | recall |")
            lines.append("|---:|---:|---:|---:|---:|---:|")
            for row in first_with_sweep["generation"]["tau_low_sweep"]:
                lines.append(
                    "| {t:.2f} | {tp} | {fp} | {fn} | {p} | {r} |".format(
                        t=row["tau_low"],
                        tp=row["floor_tp"],
                        fp=row["floor_fp"],
                        fn=row["floor_fn"],
                        p=_format_pct(row["floor_precision"]) if row["floor_precision"] is not None else "-",
                        r=_format_pct(row["floor_recall"]) if row["floor_recall"] is not None else "-",
                    )
                )
            lines.append("")
    else:
        lines.append(
            "No `--full` runs yet. Once `OPENAI_API_KEY` is set, run "
            "`python -m evals.run --config experiments/<name>.yaml --full` "
            "to populate this section. The eval ships an OpenAI-compatible "
            "generator (`generator.py`) and an LLM-as-judge "
            "(`evals/judge.py`); both honour `OPENAI_BASE_URL` and the YAML's "
            "`generation.base_url` so any provider works (see DEFERRED.md)."
        )
        lines.append("")

    return "\n".join(lines)


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
                      help="Run retrieval metrics only (hit@k, MRR). Default mode.")
    mode.add_argument("--full", action="store_true",
                      help="Also run generation, LLM-judge, ROUGE-L, refusal P/R. "
                           "Requires OPENAI_API_KEY; uses OPENAI_BASE_URL "
                           "or YAML generation.base_url for non-OpenAI providers.")
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
