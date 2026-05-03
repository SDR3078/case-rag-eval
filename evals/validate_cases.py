"""Sanity-check the evaluation case file against the persisted index.

Run from the project root with `.venv/bin/python evals/validate_cases.py`.
Exits non-zero on any schema or referential-integrity failure.
"""
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CASES = ROOT / "evals" / "cases.jsonl"
CHUNKS = ROOT / "index" / "default" / "chunks.jsonl"

REQUIRED_KEYS = {
    "id", "category", "question", "expected_chunk_ids",
    "expected_answer_themes", "expected_behaviour", "notes",
}
ALLOWED_CATEGORIES = {"golden_path", "multi_faq", "adversarial", "out_of_scope"}
ALLOWED_BEHAVIOURS = {"answer", "refuse"}


def main() -> int:
    """Validate cases.jsonl; return 0 on success, 1 on any failure."""
    errors: list[str] = []
    if not CHUNKS.is_file():
        print(
            f"error: {CHUNKS} not found. Build the default index first:\n"
            f"  .venv/bin/python -m ingest --config experiments/default.yaml",
            file=sys.stderr,
        )
        return 1
    chunk_ids = {json.loads(line)["id"] for line in CHUNKS.read_text().splitlines() if line.strip()}
    seen_ids: set[str] = set()
    by_category: Counter[str] = Counter()

    for n, line in enumerate(CASES.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            c = json.loads(line)
        except json.JSONDecodeError as e:
            errors.append(f"line {n}: invalid JSON ({e})")
            continue
        missing = REQUIRED_KEYS - c.keys()
        if missing:
            errors.append(f"line {n} ({c.get('id', '?')}): missing keys {missing}")
            continue
        if c["id"] in seen_ids:
            errors.append(f"line {n}: duplicate id {c['id']}")
        seen_ids.add(c["id"])
        if c["category"] not in ALLOWED_CATEGORIES:
            errors.append(f"line {n} ({c['id']}): unknown category {c['category']!r}")
        if c["expected_behaviour"] not in ALLOWED_BEHAVIOURS:
            errors.append(f"line {n} ({c['id']}): unknown behaviour {c['expected_behaviour']!r}")
        if not isinstance(c["expected_chunk_ids"], list) or not isinstance(c["expected_answer_themes"], list):
            errors.append(f"line {n} ({c['id']}): expected_chunk_ids and expected_answer_themes must be lists")
            continue
        if c["expected_behaviour"] == "answer" and not c["expected_chunk_ids"]:
            errors.append(f"line {n} ({c['id']}): answer cases must have non-empty expected_chunk_ids")
        if c["expected_behaviour"] == "refuse" and c["expected_chunk_ids"]:
            errors.append(f"line {n} ({c['id']}): refuse cases must have empty expected_chunk_ids")
        for cid in c["expected_chunk_ids"]:
            if cid not in chunk_ids:
                errors.append(f"line {n} ({c['id']}): unknown chunk id {cid!r} (not in {CHUNKS})")
        by_category[c["category"]] += 1

    if errors:
        print("VALIDATION FAILED", file=sys.stderr)
        for err in errors:
            print(f"  - {err}", file=sys.stderr)
        return 1

    print(f"OK — {sum(by_category.values())} cases validated against {len(chunk_ids)} chunks")
    for cat in sorted(by_category):
        print(f"  {cat:15s} {by_category[cat]:3d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
