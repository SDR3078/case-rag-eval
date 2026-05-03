"""ROUGE-L F-measure as a string-overlap baseline against canonical FAQ answers.

ROUGE-L is a cheap unsupervised reference for the LLM-as-judge correctness
score: it cannot tell faithfulness from boilerplate restatement, but it
catches obvious off-topic generations and gives the eval table a non-API
column.
"""

from __future__ import annotations

from rouge_score import rouge_scorer

# `use_stemmer=True` makes "report"/"reporting" match — useful on regulatory
# prose where verb tenses vary widely.
_SCORER = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)


def rouge_l(answer: str, reference: str) -> float:
    """Return the ROUGE-L F-measure between `answer` and `reference` in [0, 1].

    Empty inputs return 0.0 rather than raising.
    """
    if not isinstance(answer, str) or not isinstance(reference, str):
        raise TypeError("answer and reference must be strings")
    if not answer.strip() or not reference.strip():
        return 0.0
    return float(_SCORER.score(reference, answer)["rougeL"].fmeasure)


def rouge_l_against_chunks(answer: str, chunk_texts: list[str]) -> float:
    """ROUGE-L of `answer` against the concatenation of all `chunk_texts`.

    Used for multi-FAQ cases where the canonical answer spans several chunks.
    Concatenating biases recall slightly upward (more reference text to match
    against) but is a reasonable cheap baseline; LLM-judge correctness is
    the primary metric, ROUGE-L is just a sanity-check column.
    """
    reference = "\n\n".join(chunk_texts)
    return rouge_l(answer, reference)
