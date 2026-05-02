"""Optional cross-encoder reranker.

Wraps a sentence-transformers `CrossEncoder` (default: `BAAI/bge-reranker-v2-m3`)
behind a tiny `Reranker` interface. When the configured model is `None`, the
reranker is a noop and returns the input chunks unchanged so the rest of the
pipeline doesn't have to special-case its absence.
"""

from __future__ import annotations

from retriever import RetrievalResult


class Reranker:
    """Score `(query, chunk_text)` pairs with a cross-encoder and re-sort.

    `model_name=None` puts the reranker into noop mode. Otherwise the constructor
    eagerly loads the cross-encoder so the first query doesn't pay the load cost.
    """

    def __init__(self, model_name: str | None):
        self.model_name = model_name
        if model_name is None:
            self._model = None
        else:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(model_name)

    def rerank(
        self, query: str, results: list[RetrievalResult], top_k: int
    ) -> list[RetrievalResult]:
        """Return the `top_k` highest-scoring chunks for the query.

        In noop mode this is `results[:top_k]`. With a real model, scores are
        cross-encoder logits (higher = more relevant); we replace each result's
        `score` with the rerank score and re-tag the method.
        """
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        if not isinstance(top_k, int) or top_k < 1:
            raise ValueError("top_k must be a positive int")
        if not results:
            return []

        if self._model is None:
            return results[:top_k]

        pairs = [(query, r.chunk.text) for r in results]
        scores = self._model.predict(pairs, show_progress_bar=False)
        ranked = sorted(zip(results, scores), key=lambda t: float(t[1]), reverse=True)
        out: list[RetrievalResult] = []
        for r, s in ranked[:top_k]:
            out.append(
                RetrievalResult(chunk=r.chunk, score=float(s), method="rerank")
            )
        return out
