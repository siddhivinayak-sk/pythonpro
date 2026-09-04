"""RagService: the application-facing façade over the RAG core.

Owns the vector store + index store, caches embedders per profile, and exposes retrieval + indexing used
by the FastAPI layer. Ensures the query path uses the same embedder the collection was indexed with.
"""

from __future__ import annotations

from collections.abc import Callable

from ai_agent_core import get_logger
from ai_agent_core.schemas import Citation, Hit, RetrieveRequest, RetrieveResponse

from .config import EmbeddingProfile, RagConfig, RagSettings
from .embeddings import Embedder, build_embedder
from .fusion import fuse
from .indexing import Indexer, IndexResult, IndexStore
from .rerank import Reranker, build_reranker
from .sparse import SparseRetriever, build_sparse_retriever
from .vectorstore import VectorStore, build_vector_store

log = get_logger("rag-service")


class RagService:
    def __init__(
        self,
        config: RagConfig,
        settings: RagSettings,
        *,
        vector_store: VectorStore | None = None,
        embedder_factory: Callable[[EmbeddingProfile], Embedder] = build_embedder,
        sparse_retriever: SparseRetriever | None = None,
        reranker: Reranker | None = None,
    ) -> None:
        self.config = config
        self.settings = settings
        self.store = IndexStore(settings.index_db_path)
        self.vector_store = vector_store or build_vector_store(config.vector_store)
        self.sparse = sparse_retriever or build_sparse_retriever(config)
        self.reranker = reranker or build_reranker(
            config.retrieval.rerank_model, device=config.retrieval.rerank_device
        )
        self._embedder_factory = embedder_factory
        self._embedders: dict[str, Embedder] = {}
        self.indexer = Indexer(
            config,
            self.store,
            self.vector_store,
            embedder_factory=embedder_factory,
            use_docling=settings.enable_docling,
            sparse_index=self.sparse,
        )

    def _embedder_for(self, collection: str) -> Embedder:
        spec = self.config.collection(collection)
        if spec.embedding_ref not in self._embedders:
            self._embedders[spec.embedding_ref] = self._embedder_factory(
                self.config.embedding(spec.embedding_ref)
            )
        return self._embedders[spec.embedding_ref]

    def list_collections(self) -> list[dict]:
        collections = []
        for spec in self.config.collections:
            try:
                count = self.vector_store.count(spec.name)
            except Exception:  # noqa: BLE001 - store may not be reachable/initialised yet
                count = None
            collections.append(
                {
                    "name": spec.name,
                    "embedding": spec.embedding_ref,
                    "dimension": spec.dimension,
                    "chunker": spec.chunker_ref,
                    "distance": spec.distance,
                    "hybrid": spec.hybrid,
                    "count": count,
                }
            )
        return collections

    def retrieve(self, request: RetrieveRequest) -> RetrieveResponse:
        self.config.collection(request.collection)  # raises KeyError -> 404 at the API layer
        embedder = self._embedder_for(request.collection)
        retrieval = self.config.retrieval

        k = request.k if request.k is not None else retrieval.default_k
        do_hybrid = request.mode == "hybrid"
        do_rerank = bool(request.rerank or retrieval.rerank_enabled)
        # Fetch a larger candidate pool when fusing or reranking, then narrow to k.
        fetch = max(k, retrieval.candidates) if (do_hybrid or do_rerank) else k

        query_vector = embedder.embed_query(request.query)
        dense_hits = self.vector_store.search(
            request.collection, query_vector, fetch, request.filters
        )
        if do_hybrid:
            sparse_hits = self.sparse.search(
                request.collection, request.query, fetch, request.filters
            )
            hits = fuse(dense_hits, sparse_hits, top_k=fetch)
        else:
            hits = list(dense_hits)

        hits = self.reranker.rerank(request.query, hits, k) if do_rerank else hits[:k]

        return RetrieveResponse(
            hits=[
                Hit(
                    text=h.text,
                    score=h.score,
                    citation=Citation(
                        path=h.metadata.get("path"),
                        page=h.metadata.get("page"),
                        title=h.metadata.get("title") or h.metadata.get("section"),
                    ),
                    chunk_id=h.chunk_id,
                    source_id=h.source_id,
                )
                for h in hits
            ],
            used={
                "embedding": embedder.id,
                "backend": self.config.vector_store.backend,
                "mode": "hybrid" if do_hybrid else "dense",
                "reranked": do_rerank,
            },
        )

    def run_index(self, collection: str, mode: str = "incremental") -> IndexResult:
        return self.indexer.index_collection(collection, mode)

    def index_status(self, collection: str) -> dict:
        self.config.collection(collection)  # validate name
        try:
            count = self.vector_store.count(collection)
        except Exception:  # noqa: BLE001
            count = None
        return {
            "collection": collection,
            "vectors": count,
            "last_run": self.store.last_run(collection),
        }
