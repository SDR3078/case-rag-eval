"""Retrieval over the persisted RAG index.

Loads chunks + dense embeddings + BM25 from `index/{config_name}/` and exposes
`Retriever.retrieve(query, k, method)` for dense / BM25 / hybrid (RRF) search.
The retriever owns the per-backend query-prefix asymmetry by routing through the
embedding backend's `embed_queries` method.
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ingest import Chunk, get_embedding_backend, tokenise_for_bm25

PROJECT_ROOT = Path(__file__).resolve().parent
INDEX_ROOT = PROJECT_ROOT / "index"


# --- Result type -----------------------------------------------------------


@dataclass
class RetrievalResult:
    """A retrieved chunk with the score that ranked it.

    `score` semantics depend on `method`: cosine similarity for `dense` (in
    [-1, 1], typically 0.2–0.8 in practice), BM25 raw score for `bm25`, and
    fused RRF score for `hybrid`. For hybrid the absolute number is not
    meaningful — only the rank is.
    """

    chunk: Chunk
    score: float
    method: str


# --- Loading ---------------------------------------------------------------


def _load_chunks(jsonl_path: Path) -> list[Chunk]:
    chunks: list[Chunk] = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            chunks.append(Chunk(**d))
    return chunks


# --- Retriever -------------------------------------------------------------


class Retriever:
    """Loads a persisted index and runs retrieval against it.

    Construction loads chunks, dense embeddings, BM25, and the embedding backend
    described in the persisted config. The backend is needed to embed user queries
    with the correct query-side prefix.
    """

    def __init__(self, config_name: str):
        if not isinstance(config_name, str) or not config_name.strip():
            raise ValueError("config_name must be a non-empty string")
        idx_dir = INDEX_ROOT / config_name
        if not idx_dir.is_dir():
            raise FileNotFoundError(
                f"Index dir not found: {idx_dir}. Run "
                f"`python -m ingest --config experiments/{config_name}.yaml` first."
            )

        self.config_name = config_name
        self.config = json.loads((idx_dir / "config.json").read_text(encoding="utf-8"))
        self.chunks: list[Chunk] = _load_chunks(idx_dir / "chunks.jsonl")
        self.embeddings: np.ndarray = np.load(idx_dir / "embeddings.npz")["embeddings"]
        with (idx_dir / "bm25.pkl").open("rb") as f:
            self.bm25 = pickle.load(f)

        # Backend: needed for query embedding (passages are already on disk).
        self._backend = get_embedding_backend(self.config["embedding"])

        if self.embeddings.shape[0] != len(self.chunks):
            raise RuntimeError(
                f"Index inconsistent: {self.embeddings.shape[0]} embeddings vs "
                f"{len(self.chunks)} chunks in {idx_dir}"
            )

        self._chunks_by_id = {c.id: c for c in self.chunks}

    # -- public ----------------------------------------------------------

    def retrieve(
        self, query: str, k: int = 5, method: str = "dense"
    ) -> list[RetrievalResult]:
        """Return the top-`k` chunks for `query` using the configured method.

        Methods:
          - `"dense"`: cosine similarity over the persisted embedding matrix.
          - `"bm25"`: BM25Okapi raw score over the same chunks.
          - `"hybrid"`: dense top-20 + BM25 top-20, fused with RRF (k=60), top-`k`.

        After retrieval, results are deduplicated by `parent_id` (when present)
        so that split-window chunks for the same FAQ entry don't take multiple
        slots in the top-k.
        """
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        if not isinstance(k, int) or k < 1:
            raise ValueError("k must be a positive int")

        if method == "dense":
            results = self._dense(query, k * 4)  # over-retrieve before dedupe
        elif method == "bm25":
            results = self._bm25(query, k * 4)
        elif method == "hybrid":
            results = self._hybrid(query, k * 4)
        else:
            raise ValueError(f"Unknown retrieval method: {method!r}")

        deduped = self._dedupe_by_parent(results)
        return deduped[:k]

    # -- top-k helper ----------------------------------------------------

    def _wrap_top_k(
        self, scores: np.ndarray, k: int, method: str
    ) -> list[RetrievalResult]:
        """Pick the top-`k` indices by score and wrap them as RetrievalResults.

        Caps `k` at `len(scores)` so the function is robust to over-retrieve
        factors larger than the corpus.
        """
        k = min(k, len(scores))
        top_idx = np.argpartition(-scores, k - 1)[:k]
        top_idx = top_idx[np.argsort(-scores[top_idx])]
        return [
            RetrievalResult(chunk=self.chunks[i], score=float(scores[i]), method=method)
            for i in top_idx
        ]

    # -- dense -----------------------------------------------------------

    def _dense(self, query: str, k: int) -> list[RetrievalResult]:
        q_vec = self._backend.embed_queries([query])[0]
        # Embeddings are already L2-normalised at ingest time, so dot product = cosine.
        scores = self.embeddings @ q_vec
        return self._wrap_top_k(scores, k, "dense")

    # -- BM25 ------------------------------------------------------------

    def _bm25(self, query: str, k: int) -> list[RetrievalResult]:
        toks = tokenise_for_bm25(query)
        scores = np.asarray(self.bm25.get_scores(toks))
        return self._wrap_top_k(scores, k, "bm25")

    # -- hybrid (RRF) ----------------------------------------------------

    _RRF_K = 60

    def _hybrid(self, query: str, k: int) -> list[RetrievalResult]:
        dense = self._dense(query, 20)
        sparse = self._bm25(query, 20)
        fused: dict[str, float] = {}
        # RRF assigns 1 / (k + rank), summed over rankers.
        for rank, r in enumerate(dense):
            fused[r.chunk.id] = fused.get(r.chunk.id, 0.0) + 1.0 / (self._RRF_K + rank + 1)
        for rank, r in enumerate(sparse):
            fused[r.chunk.id] = fused.get(r.chunk.id, 0.0) + 1.0 / (self._RRF_K + rank + 1)
        sorted_ids = sorted(fused, key=fused.get, reverse=True)[:k]
        return [
            RetrievalResult(chunk=self._chunks_by_id[i], score=fused[i], method="hybrid")
            for i in sorted_ids
        ]

    # -- dedupe ----------------------------------------------------------

    @staticmethod
    def _dedupe_by_parent(
        results: list[RetrievalResult],
    ) -> list[RetrievalResult]:
        """Keep the highest-scoring window per parent_id; pass non-parented through."""
        seen_parents: set[str] = set()
        out: list[RetrievalResult] = []
        for r in results:
            pid = r.chunk.parent_id
            if pid is None:
                out.append(r)
            elif pid not in seen_parents:
                seen_parents.add(pid)
                out.append(r)
        return out
