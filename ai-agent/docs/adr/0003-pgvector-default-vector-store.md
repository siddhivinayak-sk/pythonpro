# ADR 0003 — pgvector as the default vector store

- **Status:** Accepted (2026-09)
- **Context:** The RAG app must support multiple vector DBs (Milvus, PGVector, other OSS) and needs a
  sensible default that balances operability and capability.
- **Decision:** Default to **pgvector**: vectors, metadata, and joins in one ACID database, minimal ops for
  the initial scale target. Provide a `VectorStore` abstraction with adapters for `memory` (tests/dev),
  `pgvector` (default), and `chroma`; Qdrant/Milvus are reserved for higher scale.
- **Consequences:** Small/medium deployments run one datastore. At larger scale (10M+ vectors) switch the
  backend via config to Milvus/Qdrant. The dependency-free `memory` backend keeps tests fast and offline;
  the pgvector path requires a live DB and is validated via the Docker `rag` profile.
