"""The indexing pipeline: scan → detect changes → load → chunk → embed → upsert.

Incremental mode skips files whose content hash is unchanged and removes vectors for deleted files, so
re-runs are cheap and idempotent. Full mode re-embeds everything. Components are injected for testing.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import asdict, dataclass

from ai_agent_core import get_logger

from ..chunking import Chunker, build_chunker
from ..config import EmbeddingProfile, RagConfig
from ..embeddings import Embedder, build_embedder
from ..loaders import LoaderRegistry, build_default_registry
from ..vectorstore import VectorStore
from .scanner import scan_directory
from .store import IndexStore

log = get_logger("rag-indexer")


@dataclass
class IndexResult:
    collection: str
    mode: str
    scanned: int
    changed: int
    deleted: int
    failed: int
    vectors_upserted: int

    def to_dict(self) -> dict:
        return asdict(self)


class Indexer:
    def __init__(
        self,
        config: RagConfig,
        store: IndexStore,
        vector_store: VectorStore,
        *,
        loader_registry: LoaderRegistry | None = None,
        embedder_factory: Callable[[EmbeddingProfile], Embedder] = build_embedder,
        chunker_factory: Callable = build_chunker,
        use_docling: bool = False,
    ) -> None:
        self.config = config
        self.store = store
        self.vector_store = vector_store
        self.loaders = loader_registry or build_default_registry(use_docling=use_docling)
        self._embedder_factory = embedder_factory
        self._chunker_factory = chunker_factory

    def index_collection(self, name: str, mode: str = "incremental") -> IndexResult:
        started = time.time()
        spec = self.config.collection(name)

        embedding_profile = self.config.embedding(spec.embedding_ref)
        if embedding_profile.dimension != spec.dimension:
            raise ValueError(
                f"collection '{name}' dimension {spec.dimension} != embedding profile "
                f"'{embedding_profile.id}' dimension {embedding_profile.dimension}"
            )
        embedder = self._embedder_factory(embedding_profile)
        if embedder.dimension != spec.dimension:
            raise ValueError(
                f"embedder produced dimension {embedder.dimension}, expected {spec.dimension}"
            )
        chunker: Chunker = self._chunker_factory(self.config.chunker(spec.chunker_ref))

        self.vector_store.ensure_collection(name, spec.dimension, spec.distance)

        supported = self.loaders.supported_extensions()
        current = {}
        for source in self.config.sources_for(name):
            for scanned_file in scan_directory(source, supported_extensions=supported):
                current[scanned_file.source_id] = scanned_file

        deleted = 0
        for stale_id in self.store.list_source_ids(name) - set(current):
            self.vector_store.delete_by_source(name, stale_id)
            self.store.delete_file(name, stale_id)
            deleted += 1

        changed = failed = upserted = 0
        for source_id, scanned_file in current.items():
            existing = self.store.get_file(name, source_id)
            if (
                mode == "incremental"
                and existing
                and existing.content_hash == scanned_file.content_hash
            ):
                continue

            loader = self.loaders.for_path(scanned_file.path)
            if loader is None:
                continue
            try:
                doc = loader.load(scanned_file.path, source_id)
                chunks = chunker.split(doc)
                self.vector_store.delete_by_source(name, source_id)  # replace prior chunks
                if chunks:
                    vectors = embedder.embed_documents([c.text for c in chunks])
                    self.vector_store.upsert(name, chunks, vectors)
                    upserted += len(chunks)
                self.store.upsert_file(
                    name,
                    source_id,
                    str(scanned_file.path),
                    scanned_file.content_hash,
                    scanned_file.size,
                    scanned_file.mtime,
                )
                changed += 1
            except Exception as exc:  # noqa: BLE001 - one bad file shouldn't fail the whole run
                failed += 1
                log.warning("index_file_failed", source_id=source_id, error=str(exc))

        result = IndexResult(
            collection=name,
            mode=mode,
            scanned=len(current),
            changed=changed,
            deleted=deleted,
            failed=failed,
            vectors_upserted=upserted,
        )
        self.store.record_run(
            name,
            mode,
            started_at=started,
            scanned=result.scanned,
            changed=changed,
            deleted=deleted,
            failed=failed,
            vectors_upserted=upserted,
        )
        log.info("index_complete", **result.to_dict())
        return result
