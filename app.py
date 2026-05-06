"""CLI entrypoint for the EU Taxonomy FAQ RAG chatbot.

Loads the default config (or a `--config` override), runs one round-trip
(retrieve → optional rerank → refusal-floor check → generate) and prints the
answer plus the cited FAQ headings to stdout.

Usage:
    python -m app "What does the EU Taxonomy require for Article 8 reporting?"
    python -m app --config experiments/bge_large_hybrid_rerank.yaml "Question..."
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

from generator import CANONICAL_REFUSAL, Generator
from reranker import Reranker
from retriever import Retriever

PROJECT_ROOT = Path(__file__).resolve().parent

MAX_QUERY_CHARS = 500


def _load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _validate_query(query: str) -> str:
    """Trim, length-cap, and reject empty queries.

    The architecture caps user input at 500 characters to keep retrieval input
    well-behaved (and to make accidental paste-bombs into the CLI harmless).
    """
    if not isinstance(query, str):
        raise TypeError("query must be a string")
    q = query.strip()
    if not q:
        raise ValueError("query is empty")
    if len(q) > MAX_QUERY_CHARS:
        raise ValueError(f"query exceeds {MAX_QUERY_CHARS} characters; please shorten it")
    return q


def _build_pipeline(config: dict):
    """Construct the runtime pipeline objects from a config dict.

    Returns `(retriever, reranker, generator)`. The retriever loads
    its index from `index/{config.name}/`, which must already have been built
    by `python -m ingest --config <yaml>`.
    """
    name = config["name"]
    retriever = Retriever(name)
    reranker = Reranker(config["reranker"]["model"])
    gen_cfg = config["generation"]
    generator = Generator(
        model=gen_cfg["model"],
        prompt_path=config["prompt"]["path"],
        max_tokens=gen_cfg.get("max_tokens", 600),
        temperature=gen_cfg.get("temperature", 0.0),
        base_url=gen_cfg.get("base_url"),
    )
    return retriever, reranker, generator


def answer_question(query: str, config: dict) -> dict:
    """Run a single Q&A round-trip end-to-end.

    Returns a dict with `answer`, `cited_chunks` (list of `(section, question)`
    tuples), `top1_score`, `refused_at_floor` (bool), and `usage` (token info,
    or `None` if the floor short-circuited the LLM call).

    Side effects: one chat-completion API call (unless the refusal floor triggers).
    """
    query = _validate_query(query)
    retriever, reranker, generator = _build_pipeline(config)

    retrieval_cfg = config["retrieval"]
    method = retrieval_cfg.get("method", "dense")
    k_pre = retrieval_cfg.get("k_pre_rerank", 20)
    k_final = retrieval_cfg.get("k", 5)
    tau_low = float(retrieval_cfg.get("tau_low", 0.30))

    retrieved = retriever.retrieve(query, k=k_pre, method=method)
    top1_score = retrieved[0].score if retrieved else 0.0

    # Refusal floor — only meaningful for dense and dense-component scores.
    # For BM25 and hybrid (RRF) the absolute score isn't bounded the same way,
    # so the floor only applies to dense retrieval.
    if method == "dense" and top1_score < tau_low:
        return {
            "answer": CANONICAL_REFUSAL,
            "cited_chunks": [],
            "top1_score": top1_score,
            "refused_at_floor": True,
            "usage": None,
        }

    rerank_top_k = config["reranker"].get("top_k", k_final)
    reranked = reranker.rerank(query, retrieved, top_k=rerank_top_k)
    final = reranked[:k_final]

    out = generator.generate(query, final)
    return {
        "answer": out["answer"],
        "cited_chunks": [(r.chunk.section, r.chunk.question) for r in final],
        "top1_score": top1_score,
        "refused_at_floor": False,
        "usage": out["usage"],
    }


def _print_result(result: dict) -> None:
    print(result["answer"])
    print()
    if result["refused_at_floor"]:
        print(f"[refused at retrieval floor; top1 similarity {result['top1_score']:.3f}]")
        return
    print("Retrieved FAQ entries:")
    for section, question in result["cited_chunks"]:
        print(f'  - "{section}" / "{question}"')


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Ask a question of the EU Taxonomy FAQ chatbot."
    )
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "experiments" / "default.yaml"),
        help="Path to an experiment YAML (default: experiments/default.yaml)",
    )
    parser.add_argument("question", help="The user question (quoted).")
    args = parser.parse_args(argv)

    config = _load_yaml(args.config)
    try:
        result = answer_question(args.question, config)
    except (ValueError, TypeError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    _print_result(result)
    return 0


if __name__ == "__main__":
    sys.exit(_main())
