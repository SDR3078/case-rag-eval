"""Retrieval metrics.

Pure functions over `retrieved_ids` (an ordered list of chunk IDs returned by the
retriever) and `expected_ids` (the gold-set chunk IDs from an eval case). All IDs
are strings; the caller is responsible for mapping window IDs to their parent
FAQ ID (when a chunking strategy splits long entries) before passing them in.

Conventions:
- `hit_at_k` returns 1.0 / 0.0 for a single case (callers average across cases).
- `mode="any"` means at least one expected ID is in the top-k (recall stress).
  `mode="all"` means every expected ID is in the top-k (strict; useful for
  multi-FAQ recall@k).
- `mrr` is the reciprocal rank of the first expected ID found in the
  retrieved list, or 0.0 if none is found.

These are deliberately tiny (no numpy, no pandas) because the eval runner
loops over 80 cases and 10 configs - wall-clock is dominated by retrieval,
not metric arithmetic.
"""

from __future__ import annotations


def hit_at_k(
    retrieved_ids: list[str], expected_ids: list[str], k: int, mode: str = "any"
) -> float:
    """Return 1.0 if the top-k of retrieved_ids satisfies mode, else 0.0.

    mode="any": at least one expected_id appears in retrieved_ids[:k].
    mode="all": every expected_id appears in retrieved_ids[:k].

    Raises on invalid inputs; an empty expected_ids is invalid (callers should
    filter OOS cases before computing hit@k).
    """
    if not isinstance(retrieved_ids, list):
        raise TypeError("retrieved_ids must be a list")
    if not isinstance(expected_ids, list):
        raise TypeError("expected_ids must be a list")
    if not isinstance(k, int) or k < 1:
        raise ValueError("k must be a positive int")
    if not expected_ids:
        raise ValueError("expected_ids must be non-empty (filter OOS cases first)")
    if mode not in ("any", "all"):
        raise ValueError("mode must be 'any' or 'all'")

    top = set(retrieved_ids[:k])
    if mode == "any":
        return 1.0 if any(e in top for e in expected_ids) else 0.0
    return 1.0 if all(e in top for e in expected_ids) else 0.0


def mrr(retrieved_ids: list[str], expected_ids: list[str]) -> float:
    """Reciprocal rank of the first expected_id found in retrieved_ids.

    Returns 0.0 if no expected ID appears anywhere in retrieved_ids. For
    multi-FAQ cases (multiple expected IDs), this rewards surfacing any of
    the gold chunks early - it is not a per-expected-ID average. That choice
    keeps MRR comparable across single- and multi-FAQ cases on the same scale.
    """
    if not isinstance(retrieved_ids, list):
        raise TypeError("retrieved_ids must be a list")
    if not isinstance(expected_ids, list):
        raise TypeError("expected_ids must be a list")
    if not expected_ids:
        raise ValueError("expected_ids must be non-empty (filter OOS cases first)")

    expected = set(expected_ids)
    for i, rid in enumerate(retrieved_ids):
        if rid in expected:
            return 1.0 / (i + 1)
    return 0.0


# --- Smoke check -----------------------------------------------------------

if __name__ == "__main__":
    # Single-target hit@k.
    assert hit_at_k(["a", "b", "c"], ["a"], 1) == 1.0
    assert hit_at_k(["a", "b", "c"], ["a"], 3) == 1.0
    assert hit_at_k(["a", "b", "c"], ["d"], 3) == 0.0
    assert hit_at_k(["a", "b", "c"], ["c"], 1) == 0.0
    assert hit_at_k(["a", "b", "c"], ["c"], 3) == 1.0

    # Multi-target hit@k with both modes.
    assert hit_at_k(["a", "b", "c"], ["a", "b"], 5, mode="any") == 1.0
    assert hit_at_k(["a", "b", "c"], ["a", "b"], 5, mode="all") == 1.0
    assert hit_at_k(["a", "b", "c"], ["a", "z"], 5, mode="any") == 1.0
    assert hit_at_k(["a", "b", "c"], ["a", "z"], 5, mode="all") == 0.0
    assert hit_at_k(["a", "b", "c", "d", "e"], ["a", "e"], 5, mode="all") == 1.0
    assert hit_at_k(["a", "b", "c", "d", "e"], ["a", "e"], 4, mode="all") == 0.0

    # MRR.
    assert mrr(["a", "b", "c"], ["a"]) == 1.0
    assert mrr(["a", "b", "c"], ["b"]) == 0.5
    assert abs(mrr(["a", "b", "c"], ["c"]) - 1.0 / 3) < 1e-12
    assert mrr(["a", "b", "c"], ["d"]) == 0.0
    # MRR with multi-target picks earliest match.
    assert mrr(["a", "b", "c"], ["c", "a"]) == 1.0
    assert abs(mrr(["a", "b", "c"], ["c", "z"]) - 1.0 / 3) < 1e-12

    # Validation.
    try:
        hit_at_k(["a"], [], 1)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError on empty expected_ids")

    try:
        hit_at_k(["a"], ["a"], 0)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError on k<1")

    try:
        hit_at_k(["a"], ["a"], 1, mode="bogus")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError on bogus mode")

    print("metrics.py: smoke check OK")
