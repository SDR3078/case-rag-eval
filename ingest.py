"""Build the RAG index for the EU Taxonomy FAQ chatbot.

Loads `docs/taxonomy_faqs_cleaned.md`, chunks it per a configurable strategy,
embeds each chunk with the configured backend (BGE-small/-large or Voyage),
and persists chunks + dense embeddings + a BM25 index to `index/{name}/`.

Public entrypoint: `build_index(config: dict) -> None`. Idempotent — running it
twice with the same config rebuilds the index in place.

CLI:
    python -m ingest --config experiments/default.yaml
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import yaml

# --- Project root ----------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
INDEX_ROOT = PROJECT_ROOT / "index"


# --- Chunk dataclass -------------------------------------------------------


@dataclass
class Chunk:
    """One retrieval unit. `text` is what gets embedded and shown to the model."""

    id: str
    section: str
    question: str
    text: str
    token_count: int
    parent_id: str | None = None  # for split-window chunks (Variant B)


# --- Markdown parsing ------------------------------------------------------


def _parse_faq_markdown(md_path: Path) -> list[tuple[str, str, str]]:
    """Parse the FAQ markdown into `(section, question, body)` triples.

    Walks the file linearly. A `## Heading` line opens a section. A `### Heading`
    line opens a question whose body runs until the next `###` or `##`.

    Returns the list of triples in document order. Raises if the file is missing
    or contains a `###` before any `##`.
    """
    if not md_path.is_file():
        raise FileNotFoundError(f"FAQ markdown not found: {md_path}")

    text = md_path.read_text(encoding="utf-8")
    lines = text.splitlines()

    triples: list[tuple[str, str, str]] = []
    section: str | None = None
    question: str | None = None
    body_lines: list[str] = []

    def _flush():
        if section is not None and question is not None:
            body = "\n".join(body_lines).strip()
            triples.append((section, question, body))

    for raw in lines:
        line = raw.rstrip()
        if line.startswith("## ") and not line.startswith("### "):
            _flush()
            question = None
            body_lines = []
            section = line[3:].strip()
        elif line.startswith("### "):
            _flush()
            if section is None:
                raise ValueError(
                    f"Found '### {line[4:].strip()}' before any '##' section heading"
                )
            question = line[4:].strip()
            body_lines = []
        else:
            if question is not None:
                body_lines.append(raw)

    _flush()
    return triples


# --- Token counting --------------------------------------------------------


def _approx_token_count(s: str) -> int:
    """Cheap whitespace-based token estimate.

    Used only for the long-entry split decision and bookkeeping. Roughly matches
    BPE token counts to within ~25% for English prose, which is more than enough
    to trigger the 800-token outlier cutoff.
    """
    return len(s.split())


# --- ID generation ---------------------------------------------------------


def _stable_id(section: str, question: str, suffix: str = "") -> str:
    """Deterministic short ID. Same (section, question) pair always produces the same ID.

    The hash makes the IDs stable across reruns and across machines, which matters
    for the experiment matrix where eval cases reference chunk IDs by name.
    """
    h = hashlib.sha1(f"{section}|||{question}|||{suffix}".encode("utf-8")).hexdigest()
    return f"faq_{h[:10]}"


# --- Chunking strategies ---------------------------------------------------


def _chunk_per_question(
    triples: list[tuple[str, str, str]],
) -> list[Chunk]:
    """Default strategy: one chunk per `### Question`, no section header in text."""
    chunks: list[Chunk] = []
    for section, question, body in triples:
        text = f"### {question}\n\n{body}".strip()
        chunks.append(
            Chunk(
                id=_stable_id(section, question),
                section=section,
                question=question,
                text=text,
                token_count=_approx_token_count(text),
            )
        )
    return chunks


def _chunk_per_question_with_section(
    triples: list[tuple[str, str, str]],
) -> list[Chunk]:
    """Variant A: prepend the parent `## Section` heading to the chunk text.

    The metadata is unchanged; only the text payload (which the embedding model
    sees) gains a section prior. See §1 of the architecture doc.
    """
    chunks: list[Chunk] = []
    for section, question, body in triples:
        text = f"# {section}\n\n### {question}\n\n{body}".strip()
        chunks.append(
            Chunk(
                id=_stable_id(section, question),
                section=section,
                question=question,
                text=text,
                token_count=_approx_token_count(text),
            )
        )
    return chunks


def _split_long(
    chunks: list[Chunk],
    threshold_tokens: int,
    window_tokens: int,
    overlap_tokens: int,
) -> list[Chunk]:
    """Variant B: sliding-window split for any chunk over `threshold_tokens`.

    Short chunks are passed through untouched. A long chunk becomes N overlapping
    windows that share the original chunk's `id` as `parent_id`. The retriever
    dedupes by `parent_id` at query time.
    """
    if window_tokens <= overlap_tokens:
        raise ValueError("window_tokens must be greater than overlap_tokens")

    out: list[Chunk] = []
    for c in chunks:
        if c.token_count <= threshold_tokens:
            out.append(c)
            continue
        words = c.text.split()
        step = window_tokens - overlap_tokens
        windows = [
            words[i : i + window_tokens]
            for i in range(0, len(words), step)
            if i < len(words)
        ]
        for w_idx, w_words in enumerate(windows):
            w_text = " ".join(w_words)
            out.append(
                Chunk(
                    id=_stable_id(c.section, c.question, suffix=f"win{w_idx}"),
                    section=c.section,
                    question=c.question,
                    text=w_text,
                    token_count=_approx_token_count(w_text),
                    parent_id=c.id,
                )
            )
    return out


def _build_chunks(
    triples: list[tuple[str, str, str]], chunking_cfg: dict
) -> list[Chunk]:
    """Dispatch on the configured chunking strategy."""
    strategy = chunking_cfg.get("strategy", "per_question")
    if strategy == "per_question":
        return _chunk_per_question(triples)
    if strategy == "per_question_with_section":
        return _chunk_per_question_with_section(triples)
    if strategy == "per_question_with_section_split_long":
        base = _chunk_per_question_with_section(triples)
        return _split_long(
            base,
            threshold_tokens=int(chunking_cfg.get("split_long_threshold_tokens", 800)),
            window_tokens=int(chunking_cfg.get("split_window_tokens", 600)),
            overlap_tokens=int(chunking_cfg.get("split_overlap_tokens", 100)),
        )
    raise ValueError(f"Unknown chunking strategy: {strategy!r}")


# --- Embedding backends ----------------------------------------------------


class EmbeddingBackend:
    """Abstract embedding backend.

    Owns the query/passage prefix asymmetry: BGE prepends a query-only instruction,
    Voyage uses an `input_type` switch, etc. Callers MUST go through `embed_passages`
    and `embed_queries`, not a generic `encode`.
    """

    name: str
    dim: int

    def embed_passages(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        raise NotImplementedError

    def embed_queries(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        raise NotImplementedError


class _BGEBackend(EmbeddingBackend):
    """BGE family (bge-small / bge-large).

    BGE prepends `"Represent this sentence for searching relevant passages: "`
    to *queries* only. Passages are encoded raw. We L2-normalise on output so
    cosine similarity collapses to a dot product downstream.
    """

    QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

    def __init__(self, hf_model: str, name: str):
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(hf_model)
        self.name = name
        # `get_embedding_dimension` is the new name; older versions still expose
        # the legacy alias. Try the new one first, fall back if needed.
        get_dim = getattr(
            self._model,
            "get_embedding_dimension",
            getattr(self._model, "get_sentence_embedding_dimension", None),
        )
        self.dim = get_dim() if get_dim is not None else 0

    def embed_passages(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        return self._encode(texts, batch_size)

    def embed_queries(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        prefixed = [self.QUERY_PREFIX + t for t in texts]
        return self._encode(prefixed, batch_size)

    def _encode(self, texts: list[str], batch_size: int) -> np.ndarray:
        vecs = self._model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return vecs.astype(np.float32, copy=False)


class _VoyageBackend(EmbeddingBackend):
    """Voyage AI hosted embeddings (e.g. voyage-3-large).

    Voyage exposes an `input_type` parameter ("document" vs "query") that plays
    the same role as BGE's prefix asymmetry. Outputs are NOT pre-normalised by
    the API; we normalise here.
    """

    def __init__(self, model: str, name: str):
        try:
            import voyageai  # type: ignore
        except ImportError as e:
            raise ImportError(
                "voyageai not installed. `pip install voyageai` and set VOYAGE_API_KEY."
            ) from e
        api_key = os.environ.get("VOYAGE_API_KEY")
        if not api_key:
            raise RuntimeError("VOYAGE_API_KEY env var is required for the voyage_3_large backend")
        self._client = voyageai.Client(api_key=api_key)
        self._model = model
        self.name = name
        # Voyage-3-large is 1024-dim; we set this lazily after the first call.
        self.dim = 1024

    def embed_passages(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        return self._call(texts, input_type="document", batch_size=batch_size)

    def embed_queries(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        return self._call(texts, input_type="query", batch_size=batch_size)

    def _call(self, texts: list[str], input_type: str, batch_size: int) -> np.ndarray:
        out: list[list[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            resp = self._client.embed(batch, model=self._model, input_type=input_type)
            out.extend(resp.embeddings)
        arr = np.asarray(out, dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.dim = arr.shape[1]
        return arr / norms


_EMBEDDING_BACKEND_REGISTRY = {
    "bge_small": ("BAAI/bge-small-en-v1.5", _BGEBackend),
    "bge_large": ("BAAI/bge-large-en-v1.5", _BGEBackend),
    "voyage_3_large": ("voyage-3-large", _VoyageBackend),
}


def get_embedding_backend(backend_cfg: dict) -> EmbeddingBackend:
    """Build the configured embedding backend.

    `backend_cfg` is the `embedding:` block of an experiment config. The
    `backend` key picks the registered backend; an optional `model` key
    overrides the default HF/Voyage model id.
    """
    name = backend_cfg.get("backend")
    if name not in _EMBEDDING_BACKEND_REGISTRY:
        raise ValueError(
            f"Unknown embedding backend {name!r}. "
            f"Choose from {sorted(_EMBEDDING_BACKEND_REGISTRY)}"
        )
    default_model, cls = _EMBEDDING_BACKEND_REGISTRY[name]
    model = backend_cfg.get("model", default_model)
    return cls(model, name)


# --- BM25 ------------------------------------------------------------------


def tokenise_for_bm25(text: str) -> list[str]:
    """Lowercase and split on word characters. Cheap, deterministic.

    Public because retrieval-time tokenisation MUST match index-time tokenisation
    or BM25 scores will silently mismatch. `retriever.py` imports this directly.
    """
    return re.findall(r"\w+", text.lower())


def _build_bm25(chunks: list[Chunk]):
    from rank_bm25 import BM25Okapi

    tokenised = [tokenise_for_bm25(c.text) for c in chunks]
    return BM25Okapi(tokenised)


# --- Persistence -----------------------------------------------------------


def _index_dir(name: str) -> Path:
    return INDEX_ROOT / name


def _persist(
    name: str,
    chunks: list[Chunk],
    embeddings: np.ndarray,
    bm25,
    config: dict,
) -> None:
    out_dir = _index_dir(name)
    out_dir.mkdir(parents=True, exist_ok=True)

    with (out_dir / "chunks.jsonl").open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")

    np.savez_compressed(out_dir / "embeddings.npz", embeddings=embeddings)

    with (out_dir / "bm25.pkl").open("wb") as f:
        pickle.dump(bm25, f)

    (out_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")


# --- Config validation -----------------------------------------------------

_REQUIRED_TOP_KEYS = ("name", "corpus", "chunking", "embedding")


def _validate_config(cfg: dict) -> None:
    if not isinstance(cfg, dict):
        raise TypeError("config must be a dict")
    missing = [k for k in _REQUIRED_TOP_KEYS if k not in cfg]
    if missing:
        raise ValueError(f"config missing required keys: {missing}")
    if not isinstance(cfg["name"], str) or not cfg["name"].strip():
        raise ValueError("config.name must be a non-empty string")
    if "path" not in cfg["corpus"]:
        raise ValueError("config.corpus.path is required")
    if "strategy" not in cfg["chunking"]:
        raise ValueError("config.chunking.strategy is required")
    if "backend" not in cfg["embedding"]:
        raise ValueError("config.embedding.backend is required")


# --- Public entrypoint -----------------------------------------------------


def build_index(config: dict) -> None:
    """Build and persist the RAG index for the given experiment config.

    Side effects: writes `index/{config.name}/{chunks.jsonl, embeddings.npz, bm25.pkl, config.json}`.
    Raises on missing corpus, unknown chunking strategy, or unknown embedding backend.
    """
    _validate_config(config)
    name = config["name"]

    md_path = (PROJECT_ROOT / config["corpus"]["path"]).resolve()
    triples = _parse_faq_markdown(md_path)
    print(f"[ingest] parsed {len(triples)} FAQ entries from {md_path.name}")

    chunks = _build_chunks(triples, config["chunking"])
    print(f"[ingest] built {len(chunks)} chunks (strategy={config['chunking']['strategy']})")

    backend = get_embedding_backend(config["embedding"])
    batch_size = int(config["embedding"].get("batch_size", 32))
    print(f"[ingest] embedding backend: {backend.name} (dim={backend.dim})")
    embeddings = backend.embed_passages([c.text for c in chunks], batch_size=batch_size)
    print(f"[ingest] embedded {embeddings.shape[0]} chunks → matrix {embeddings.shape}")

    bm25 = _build_bm25(chunks)
    print(f"[ingest] built BM25 index over {len(chunks)} chunks")

    _persist(name, chunks, embeddings, bm25, config)
    print(f"[ingest] wrote index to {_index_dir(name)}")


# --- CLI -------------------------------------------------------------------


def _load_yaml(path: str | os.PathLike) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build the RAG index.")
    parser.add_argument(
        "--config", required=True, help="Path to an experiment YAML, e.g. experiments/default.yaml"
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    config = _load_yaml(args.config)
    build_index(config)


if __name__ == "__main__":
    _main()
