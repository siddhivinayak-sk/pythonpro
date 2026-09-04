"""Sparse (lexical) retrieval via a pluggable ``SparseRetriever``.

The default ``InMemoryBM25Retriever`` is a dependency-free Okapi BM25 index over chunk text, populated
during indexing alongside the vector store and queried for the sparse arm of hybrid search. It is an
in-process index (rebuilt by (re)indexing after a restart); a persistent/native sparse backend
(pgvector FTS, Qdrant/Milvus sparse vectors) is a future plug-in behind the same protocol.
"""

from __future__ import annotations

import math
import re
import threading
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from ai_agent_core import get_logger

from .filters import matches
from .models import Chunk, RetrievedHit

if TYPE_CHECKING:
    from .config import RagConfig

log = get_logger("rag-sparse")

_TOKEN = re.compile(r"[a-z0-9]+")
_K1 = 1.5
_B = 0.75


def _tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


@runtime_checkable
class SparseRetriever(Protocol):
    def index(self, collection: str, chunks: list[Chunk]) -> None: ...
    def delete_by_source(self, collection: str, source_id: str) -> None: ...
    def search(
        self, collection: str, query: str, k: int, filters: dict | None = None
    ) -> list[RetrievedHit]: ...


class _CollectionIndex:
    """Inverted index + document stats for one collection."""

    def __init__(self) -> None:
        self.postings: dict[str, dict[str, int]] = {}  # term -> {chunk_id: term_frequency}
        self.doc_len: dict[str, int] = {}
        self.chunks: dict[str, Chunk] = {}
        self.total_len = 0

    @property
    def n(self) -> int:
        return len(self.chunks)

    @property
    def avgdl(self) -> float:
        return (self.total_len / self.n) if self.n else 0.0

    def add(self, chunk: Chunk, tokens: list[str]) -> None:
        if chunk.id in self.chunks:
            self._remove_id(chunk.id)
        self.chunks[chunk.id] = chunk
        self.doc_len[chunk.id] = len(tokens)
        self.total_len += len(tokens)
        tf: dict[str, int] = {}
        for token in tokens:
            tf[token] = tf.get(token, 0) + 1
        for token, count in tf.items():
            self.postings.setdefault(token, {})[chunk.id] = count

    def _remove_id(self, chunk_id: str) -> None:
        if chunk_id not in self.chunks:
            return
        self.total_len -= self.doc_len.pop(chunk_id, 0)
        del self.chunks[chunk_id]
        for term in list(self.postings):
            docs = self.postings[term]
            if chunk_id in docs:
                del docs[chunk_id]
                if not docs:
                    del self.postings[term]

    def remove_by_source(self, source_id: str) -> None:
        for chunk_id in [cid for cid, ch in self.chunks.items() if ch.source_id == source_id]:
            self._remove_id(chunk_id)


class InMemoryBM25Retriever:
    """Dependency-free Okapi BM25 lexical retriever (thread-safe)."""

    def __init__(self, k1: float = _K1, b: float = _B) -> None:
        self._k1 = k1
        self._b = b
        self._collections: dict[str, _CollectionIndex] = {}
        self._lock = threading.Lock()

    def index(self, collection: str, chunks: list[Chunk]) -> None:
        with self._lock:
            idx = self._collections.setdefault(collection, _CollectionIndex())
            for chunk in chunks:
                idx.add(chunk, _tokenize(chunk.text))

    def delete_by_source(self, collection: str, source_id: str) -> None:
        with self._lock:
            idx = self._collections.get(collection)
            if idx is not None:
                idx.remove_by_source(source_id)

    def search(
        self, collection: str, query: str, k: int, filters: dict | None = None
    ) -> list[RetrievedHit]:
        with self._lock:
            idx = self._collections.get(collection)
            if idx is None or idx.n == 0:
                return []
            query_terms = set(_tokenize(query))
            avgdl = idx.avgdl
            n = idx.n
            candidates: set[str] = set()
            for term in query_terms:
                candidates.update(idx.postings.get(term, {}))

            scored: list[RetrievedHit] = []
            for chunk_id in candidates:
                chunk = idx.chunks[chunk_id]
                if not matches(chunk.metadata, filters):
                    continue
                dl = idx.doc_len[chunk_id]
                score = 0.0
                for term in query_terms:
                    docs = idx.postings.get(term)
                    if not docs or chunk_id not in docs:
                        continue
                    df = len(docs)
                    idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
                    freq = docs[chunk_id]
                    denom = (
                        freq + self._k1 * (1 - self._b + self._b * dl / avgdl) if avgdl else freq
                    )
                    if denom:
                        score += idf * (freq * (self._k1 + 1)) / denom
                if score > 0:
                    scored.append(
                        RetrievedHit(
                            text=chunk.text,
                            score=score,
                            chunk_id=chunk_id,
                            source_id=chunk.source_id,
                            metadata=chunk.metadata,
                        )
                    )
            scored.sort(key=lambda h: h.score, reverse=True)
            return scored[:k]


def build_sparse_retriever(config: RagConfig | None = None) -> SparseRetriever:
    """Construct the configured sparse retriever.

    ``bm25`` (default) is the dependency-free in-process index. ``pgfts`` is a persistent PostgreSQL
    full-text backend that reuses the pgvector DSN; it falls back to BM25 (with a warning) when no DSN is
    configured, so the service still boots.
    """
    if config is None:
        return InMemoryBM25Retriever()
    backend = (config.retrieval.sparse_backend or "bm25").lower()
    if backend == "pgfts":
        dsn = config.vector_store.pgvector.dsn
        if not dsn:
            log.warning("sparse_pgfts_without_dsn", detail="falling back to in-memory BM25")
            return InMemoryBM25Retriever()
        from .sparse_pg import PgFtsSparseRetriever

        return PgFtsSparseRetriever(dsn, language=config.retrieval.sparse_language)
    if backend != "bm25":
        log.warning("sparse_backend_unknown", backend=backend, detail="falling back to BM25")
    return InMemoryBM25Retriever()
