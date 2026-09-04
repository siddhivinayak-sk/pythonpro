"""RAG retrieval API (FastAPI), wired to the RAG core.

Implements collection listing, retrieval (embed → vector search → cited hits), on-demand indexing, and
index status. The pipeline itself lives in the ``ai_agent_rag`` core modules; this is the HTTP surface.
"""

from __future__ import annotations

from ai_agent_core import configure_logging, get_logger, load_config
from ai_agent_core.schemas import HealthStatus, RetrieveRequest, RetrieveResponse
from ai_agent_core.web import install_observability
from fastapi import FastAPI, HTTPException, Query

from . import __version__
from .config import RagConfig, RagSettings, default_rag_config
from .service import RagService


def build_config(settings: RagSettings) -> RagConfig:
    if settings.config_file:
        return load_config(RagConfig, settings.config_file)
    return default_rag_config(settings)


def create_app(settings: RagSettings | None = None, service: RagService | None = None) -> FastAPI:
    settings = settings or RagSettings()
    configure_logging(level=settings.log_level, json_logs=settings.log_json)
    log = get_logger("rag")

    if service is None:
        service = RagService(build_config(settings), settings)

    app = FastAPI(title="AI-Agent RAG API", version=__version__)
    app.state.service = service
    install_observability(
        app,
        service_name=settings.service_name,
        rate_limit_per_minute=settings.rate_limit_per_minute,
    )

    @app.get("/healthz", response_model=HealthStatus, tags=["health"])
    def healthz() -> HealthStatus:
        return HealthStatus(service=settings.service_name, version=__version__)

    @app.get("/readyz", tags=["health"])
    def readyz() -> dict:
        return {
            "status": "ready",
            "backend": service.config.vector_store.backend,
            "collections": len(service.config.collections),
        }

    @app.get("/v1/collections", tags=["retrieval"])
    def list_collections() -> dict:
        return {"collections": service.list_collections()}

    @app.post("/v1/retrieve", response_model=RetrieveResponse, tags=["retrieval"])
    def retrieve(request: RetrieveRequest) -> RetrieveResponse:
        try:
            return service.retrieve(request)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"unknown collection: {exc}") from exc
        except NotImplementedError as exc:
            raise HTTPException(status_code=501, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001 - surface store/embedder errors as 500
            log.warning("retrieve_failed", error=str(exc))
            raise HTTPException(status_code=500, detail=f"retrieval failed: {exc}") from exc

    @app.post("/v1/index/run", tags=["indexing"])
    def run_index(collection: str = Query(...), mode: str = Query("incremental")) -> dict:
        try:
            return service.run_index(collection, mode).to_dict()
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"unknown collection: {exc}") from exc
        except Exception as exc:  # noqa: BLE001
            log.warning("index_failed", collection=collection, error=str(exc))
            raise HTTPException(status_code=500, detail=f"index failed: {exc}") from exc

    @app.get("/v1/index/status", tags=["indexing"])
    def index_status(collection: str = Query(...)) -> dict:
        try:
            return service.index_status(collection)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"unknown collection: {exc}") from exc

    log.info(
        "rag_api_started",
        backend=service.config.vector_store.backend,
        collections=len(service.config.collections),
        port=settings.port,
    )
    return app


app = create_app()


def main() -> None:
    import uvicorn

    settings = RagSettings()
    uvicorn.run("ai_agent_rag.app:app", host=settings.host, port=settings.port)


if __name__ == "__main__":
    main()
