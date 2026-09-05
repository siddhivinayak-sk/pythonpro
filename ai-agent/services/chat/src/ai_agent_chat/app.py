"""Conversation backend API (FastAPI).

Wires auth (local admin + optional OIDC), conversations/messages, chat (sync + SSE streaming), settings,
vision uploads, and the registry-backed model picker. Everything except health + login requires a session.
The chat model is injectable (``model_provider``) so tests run with a fake model.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from uuid import uuid4

from ai_agent_core import build_langfuse_callbacks, configure_logging, get_logger
from ai_agent_core.schemas import HealthStatus, ModelInfo
from ai_agent_core.web import install_observability
from fastapi import Depends, FastAPI, File, HTTPException, Request, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from . import __version__
from .agent import AgentModelProvider, AgentOrchestrator
from .auth import AuthError, AuthService, OidcVerifier, TokenVerifier
from .cache import ResponseCache, build_registry_embedder, build_response_cache
from .chat import ChatOrchestrator, ModelProvider
from .config import ChatSettings
from .db import Attachment, Database, _now
from .llm import build_registry
from .moderation import ModerationProvider, build_moderation_provider
from .rag_client import RagClient
from .speech import SpeechError, SpeechProvider, build_speech_provider
from .store import ChatStore
from .toolbuilder import ToolBuilder, default_tool_builder
from .workflows import ModelProvider as WorkflowModelProvider
from .workflows import WorkflowEngine


# --- request bodies ----------------------------------------------------------
class LoginBody(BaseModel):
    username: str
    password: str


class OidcTokenBody(BaseModel):
    token: str


class ConversationCreate(BaseModel):
    title: str | None = None


class RenameBody(BaseModel):
    title: str


class ChatBody(BaseModel):
    text: str
    connection_id: str | None = None
    model_name: str | None = None
    temperature: float | None = None
    images: list[str] | None = None  # data URIs or URLs (vision)


class SettingsBody(BaseModel):
    data: dict[str, Any]


class WorkflowCreate(BaseModel):
    name: str = "Workflow"
    definition: dict[str, Any]


class WorkflowRunBody(BaseModel):
    inputs: dict[str, Any] = {}


class SpeechBody(BaseModel):
    text: str
    voice: str | None = None


def _user_dict(user) -> dict[str, Any]:
    return {
        "id": user.id,
        "subject": user.subject,
        "role": user.role,
        "source": user.source,
        "email": user.email,
        "display_name": user.display_name,
    }


def _token_from_request(request: Request) -> str | None:
    header = request.headers.get("authorization")
    if header and header.lower().startswith("bearer "):
        return header[7:]
    return request.cookies.get("session")


def get_current_user(request: Request):
    auth_service: AuthService = request.app.state.auth_service
    user = auth_service.resolve(_token_from_request(request))
    if user is None:
        raise HTTPException(status_code=401, detail="not authenticated")
    return user


def create_app(
    settings: ChatSettings | None = None,
    *,
    model_provider: ModelProvider | None = None,
    agent_model_provider: AgentModelProvider | None = None,
    tool_builder: ToolBuilder | None = None,
    workflow_model_provider: WorkflowModelProvider | None = None,
    speech_provider: SpeechProvider | None = None,
    moderation_provider: ModerationProvider | None = None,
    response_cache: ResponseCache | None = None,
    oidc_verifier: TokenVerifier | None = None,
) -> FastAPI:
    settings = settings or ChatSettings()
    configure_logging(level=settings.log_level, json_logs=settings.log_json)
    log = get_logger("chat")

    db = Database(settings)
    registry = build_registry(settings)
    store = ChatStore(db, settings)
    if oidc_verifier is None and settings.oidc_issuer and settings.auth_mode in ("oidc", "both"):
        oidc_verifier = OidcVerifier(settings.oidc_issuer, settings.oidc_audience)
    auth_service = AuthService(db, settings, oidc_verifier=oidc_verifier)
    moderation = moderation_provider or build_moderation_provider(settings)
    embedder = build_registry_embedder(registry) if settings.cache_mode == "semantic" else None
    cache = response_cache or build_response_cache(settings, embedder)
    tracing_callbacks = build_langfuse_callbacks(
        enabled=settings.tracing_enabled,
        public_key=settings.langfuse_public_key,
        secret_key=settings.langfuse_secret_key,
        host=settings.langfuse_host,
    )
    orchestrator = ChatOrchestrator(
        registry,
        store,
        settings,
        model_provider=model_provider,
        moderation=moderation,
        response_cache=cache,
        tracing_callbacks=tracing_callbacks,
    )
    agent = AgentOrchestrator(
        registry,
        store,
        settings,
        model_provider=agent_model_provider,
        moderation=moderation,
        tracing_callbacks=tracing_callbacks,
    )
    build_tools = tool_builder or default_tool_builder

    wf_model_provider: WorkflowModelProvider = workflow_model_provider or (
        lambda c, m, p: registry.get_chat_model(c, m, **p)
    )
    workflow_engine = WorkflowEngine(
        model_provider=wf_model_provider,
        rag_client=RagClient(settings.rag_api_base_url) if settings.rag_api_base_url else None,
    )

    app = FastAPI(title="AI-Agent Chat API", version=__version__)
    app.state.db = db
    app.state.registry = registry
    app.state.store = store
    app.state.auth_service = auth_service
    app.state.orchestrator = orchestrator
    app.state.agent = agent
    app.state.workflow_engine = workflow_engine
    speech = speech_provider or build_speech_provider(settings)
    app.state.speech = speech
    app.state.moderation = moderation
    app.state.response_cache = cache
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    install_observability(
        app,
        service_name=settings.service_name,
        rate_limit_per_minute=settings.rate_limit_per_minute,
    )

    uploads_dir = Path(settings.uploads_dir)

    # -- health --
    @app.get("/healthz", response_model=HealthStatus, tags=["health"])
    def healthz() -> HealthStatus:
        return HealthStatus(service=settings.service_name, version=__version__)

    # -- auth --
    @app.post("/v1/auth/login", tags=["auth"])
    def login(body: LoginBody, response: Response) -> dict:
        try:
            token = auth_service.login_local(body.username, body.password)
        except AuthError as exc:
            raise HTTPException(status_code=401, detail=str(exc)) from exc
        response.set_cookie("session", token, httponly=True, samesite="lax")
        return {"token": token, "user": _user_dict(auth_service.resolve(token))}

    @app.post("/v1/auth/oidc/token", tags=["auth"])
    def oidc_token(body: OidcTokenBody, response: Response) -> dict:
        try:
            token = auth_service.login_oidc(body.token)
        except AuthError as exc:
            raise HTTPException(status_code=401, detail=str(exc)) from exc
        response.set_cookie("session", token, httponly=True, samesite="lax")
        return {"token": token, "user": _user_dict(auth_service.resolve(token))}

    @app.get("/v1/auth/me", tags=["auth"])
    def me(user=Depends(get_current_user)) -> dict:
        return _user_dict(user)

    @app.post("/v1/auth/logout", status_code=204, tags=["auth"])
    def logout(request: Request, response: Response) -> Response:
        token = _token_from_request(request)
        if token:
            auth_service.logout(token)
        response.delete_cookie("session")
        return Response(status_code=204)

    # -- models --
    @app.get("/v1/models", response_model=list[ModelInfo], tags=["models"])
    def list_models(user=Depends(get_current_user)) -> list[ModelInfo]:
        return [
            ModelInfo(
                connection_id=m.connection_id,
                model_name=m.model_name,
                display_name=m.display_name,
                provider=m.provider.value,
                capabilities=[c.value for c in m.capabilities],
                dimensions=m.dimensions,
                context_window=m.context_window,
            )
            for m in registry.list_chat_models()
        ]

    # -- conversations --
    @app.get("/v1/conversations", tags=["conversations"])
    def list_conversations(user=Depends(get_current_user)) -> dict:
        return {"conversations": store.list_conversations(user.id)}

    @app.post("/v1/conversations", tags=["conversations"])
    def create_conversation(body: ConversationCreate, user=Depends(get_current_user)) -> dict:
        return store.create_conversation(user.id, body.title or "New chat")

    def _require_conversation(user, conversation_id: str) -> dict:
        conv = store.get_conversation(user.id, conversation_id)
        if conv is None:
            raise HTTPException(status_code=404, detail="conversation not found")
        return conv

    @app.get("/v1/conversations/{conversation_id}", tags=["conversations"])
    def get_conversation(conversation_id: str, user=Depends(get_current_user)) -> dict:
        return _require_conversation(user, conversation_id)

    @app.patch("/v1/conversations/{conversation_id}", tags=["conversations"])
    def rename_conversation(
        conversation_id: str, body: RenameBody, user=Depends(get_current_user)
    ) -> dict:
        _require_conversation(user, conversation_id)
        store.rename_conversation(user.id, conversation_id, body.title)
        return _require_conversation(user, conversation_id)

    @app.delete("/v1/conversations/{conversation_id}", status_code=204, tags=["conversations"])
    def delete_conversation(conversation_id: str, user=Depends(get_current_user)) -> Response:
        _require_conversation(user, conversation_id)
        store.delete_conversation(user.id, conversation_id)
        return Response(status_code=204)

    @app.get("/v1/conversations/{conversation_id}/messages", tags=["conversations"])
    def list_messages(conversation_id: str, user=Depends(get_current_user)) -> dict:
        _require_conversation(user, conversation_id)
        return {"messages": store.list_messages(conversation_id)}

    # -- chat --
    def _run_turn(user, conversation_id: str, body: ChatBody) -> dict:
        """Route through the tool-calling agent when the conversation has RAG/MCP tools selected."""
        eff = store.effective_settings(user.id, conversation_id)
        tools = build_tools(settings, eff)
        if tools:
            return agent.run_turn(
                user_id=user.id,
                conversation_id=conversation_id,
                text=body.text,
                tools=tools,
                images=body.images,
                connection_id=body.connection_id,
                model_name=body.model_name,
                temperature=body.temperature,
            )
        return orchestrator.complete(
            user_id=user.id,
            conversation_id=conversation_id,
            text=body.text,
            images=body.images,
            connection_id=body.connection_id,
            model_name=body.model_name,
            temperature=body.temperature,
        )

    @app.post("/v1/conversations/{conversation_id}/chat", tags=["chat"])
    def chat(conversation_id: str, body: ChatBody, user=Depends(get_current_user)) -> dict:
        _require_conversation(user, conversation_id)
        try:
            return _run_turn(user, conversation_id, body)
        except Exception as exc:  # noqa: BLE001 - surface model/config errors
            log.warning("chat_failed", error=str(exc))
            raise HTTPException(status_code=502, detail=f"chat failed: {exc}") from exc

    @app.post("/v1/conversations/{conversation_id}/chat/stream", tags=["chat"])
    def chat_stream(
        conversation_id: str, body: ChatBody, user=Depends(get_current_user)
    ) -> StreamingResponse:
        _require_conversation(user, conversation_id)

        eff = store.effective_settings(user.id, conversation_id)
        tools = build_tools(settings, eff)

        def event_stream():
            try:
                if tools:
                    # Tool-augmented turn: run the agent, then emit the grounded answer + citations.
                    msg = agent.run_turn(
                        user_id=user.id,
                        conversation_id=conversation_id,
                        text=body.text,
                        tools=tools,
                        images=body.images,
                        connection_id=body.connection_id,
                        model_name=body.model_name,
                        temperature=body.temperature,
                    )
                    yield f"data: {json.dumps({'delta': msg['content']})}\n\n"
                    if msg.get("citations"):
                        yield f"data: {json.dumps({'citations': msg['citations']})}\n\n"
                else:
                    for piece in orchestrator.stream(
                        user_id=user.id,
                        conversation_id=conversation_id,
                        text=body.text,
                        images=body.images,
                        connection_id=body.connection_id,
                        model_name=body.model_name,
                        temperature=body.temperature,
                    ):
                        yield f"data: {json.dumps({'delta': piece})}\n\n"
                yield "data: [DONE]\n\n"
            except Exception as exc:  # noqa: BLE001
                yield f"data: {json.dumps({'error': str(exc)})}\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    # -- settings --
    @app.get("/v1/settings", tags=["settings"])
    def get_settings(user=Depends(get_current_user)) -> dict:
        return {"settings": store.effective_settings(user.id)}

    @app.put("/v1/settings", tags=["settings"])
    def put_settings(body: SettingsBody, user=Depends(get_current_user)) -> dict:
        store.put_setting("user", user.id, body.data)
        return {"settings": store.effective_settings(user.id)}

    @app.get("/v1/conversations/{conversation_id}/settings", tags=["settings"])
    def get_conversation_settings(conversation_id: str, user=Depends(get_current_user)) -> dict:
        _require_conversation(user, conversation_id)
        return {"settings": store.effective_settings(user.id, conversation_id)}

    @app.put("/v1/conversations/{conversation_id}/settings", tags=["settings"])
    def put_conversation_settings(
        conversation_id: str, body: SettingsBody, user=Depends(get_current_user)
    ) -> dict:
        _require_conversation(user, conversation_id)
        store.put_setting("conversation", conversation_id, body.data)
        return {"settings": store.effective_settings(user.id, conversation_id)}

    # -- discovery (RAG collections + MCP servers, for the settings UI) --
    @app.get("/v1/rag/collections", tags=["discovery"])
    def list_rag_collections(user=Depends(get_current_user)) -> dict:
        if not settings.rag_api_base_url:
            return {"collections": []}
        return {"collections": RagClient(settings.rag_api_base_url).list_collections()}

    @app.get("/v1/mcp/servers", tags=["discovery"])
    def list_mcp_servers(user=Depends(get_current_user)) -> dict:
        return {"servers": settings.mcp_servers}

    # -- uploads (vision) --
    @app.post("/v1/uploads", tags=["uploads"])
    def upload(file: UploadFile = File(...), user=Depends(get_current_user)) -> dict:
        max_bytes = settings.max_upload_mb * 1024 * 1024
        uploads_dir.mkdir(parents=True, exist_ok=True)
        attachment_id = uuid4().hex
        suffix = Path(file.filename or "").suffix
        dest = uploads_dir / f"{attachment_id}{suffix}"
        size = 0
        with dest.open("wb") as out:
            while chunk := file.file.read(1024 * 1024):
                size += len(chunk)
                if size > max_bytes:
                    out.close()
                    dest.unlink(missing_ok=True)
                    raise HTTPException(status_code=413, detail="file too large")
                out.write(chunk)
        with db.session() as s:
            att = Attachment(
                id=attachment_id,
                user_id=user.id,
                kind="image",
                path=str(dest),
                mime=file.content_type or "application/octet-stream",
                size=size,
            )
            s.add(att)
            s.commit()
        return {
            "id": attachment_id,
            "url": f"/v1/uploads/{attachment_id}",
            "mime": file.content_type,
            "size": size,
        }

    @app.get("/v1/uploads/{attachment_id}", tags=["uploads"])
    def get_upload(attachment_id: str, user=Depends(get_current_user)) -> FileResponse:
        with db.session() as s:
            att = s.get(Attachment, attachment_id)
            if att is None or att.user_id != user.id:
                raise HTTPException(status_code=404, detail="not found")
            path, mime = att.path, att.mime
        return FileResponse(path, media_type=mime)

    # -- context files (ad-hoc grounding attached to a conversation) --
    @app.post("/v1/conversations/{conversation_id}/context", tags=["context"])
    def add_context(
        conversation_id: str, file: UploadFile = File(...), user=Depends(get_current_user)
    ) -> dict:
        _require_conversation(user, conversation_id)
        max_bytes = settings.max_upload_mb * 1024 * 1024
        raw = file.file.read(max_bytes + 1)
        if len(raw) > max_bytes:
            raise HTTPException(status_code=413, detail="file too large")
        text = raw.decode("utf-8", errors="replace")
        return store.add_context(conversation_id, user.id, file.filename or "context.txt", text)

    @app.get("/v1/conversations/{conversation_id}/context", tags=["context"])
    def list_context(conversation_id: str, user=Depends(get_current_user)) -> dict:
        _require_conversation(user, conversation_id)
        return {
            "context": [
                {"id": c["id"], "filename": c["filename"], "chars": len(c["text"])}
                for c in store.list_context(conversation_id)
            ]
        }

    @app.delete(
        "/v1/conversations/{conversation_id}/context/{context_id}",
        status_code=204,
        tags=["context"],
    )
    def delete_context(
        conversation_id: str, context_id: str, user=Depends(get_current_user)
    ) -> Response:
        _require_conversation(user, conversation_id)
        store.delete_context(conversation_id, context_id, user.id)
        return Response(status_code=204)

    # -- audio (STT / TTS) --
    @app.post("/v1/audio/transcribe", tags=["audio"])
    def transcribe(file: UploadFile = File(...), user=Depends(get_current_user)) -> dict:
        audio = file.file.read()
        try:
            return {"text": speech.transcribe(audio, file.content_type or "audio/webm")}
        except SpeechError as exc:
            raise HTTPException(status_code=501, detail=str(exc)) from exc

    @app.post("/v1/audio/speech", tags=["audio"])
    def synthesize(body: SpeechBody, user=Depends(get_current_user)) -> Response:
        try:
            audio, mime = speech.synthesize(body.text, body.voice or settings.tts_default_voice)
        except SpeechError as exc:
            raise HTTPException(status_code=501, detail=str(exc)) from exc
        return Response(content=audio, media_type=mime)

    # -- workflows --
    def _require_workflow(user, workflow_id: str) -> dict:
        wf = store.get_workflow(user.id, workflow_id)
        if wf is None:
            raise HTTPException(status_code=404, detail="workflow not found")
        return wf

    @app.get("/v1/workflows", tags=["workflows"])
    def list_workflows(user=Depends(get_current_user)) -> dict:
        return {"workflows": store.list_workflows(user.id)}

    @app.post("/v1/workflows", tags=["workflows"])
    def create_workflow(body: WorkflowCreate, user=Depends(get_current_user)) -> dict:
        return store.create_workflow(user.id, body.name, body.definition)

    @app.get("/v1/workflows/{workflow_id}", tags=["workflows"])
    def get_workflow(workflow_id: str, user=Depends(get_current_user)) -> dict:
        return _require_workflow(user, workflow_id)

    @app.delete("/v1/workflows/{workflow_id}", status_code=204, tags=["workflows"])
    def delete_workflow(workflow_id: str, user=Depends(get_current_user)) -> Response:
        _require_workflow(user, workflow_id)
        store.delete_workflow(user.id, workflow_id)
        return Response(status_code=204)

    @app.post("/v1/workflows/{workflow_id}/run", tags=["workflows"])
    def run_workflow(
        workflow_id: str, body: WorkflowRunBody, user=Depends(get_current_user)
    ) -> dict:
        wf = _require_workflow(user, workflow_id)
        started = _now()
        result = workflow_engine.run(wf["definition"], body.inputs)
        return store.record_run(
            workflow_id,
            user.id,
            status=result["status"],
            inputs=body.inputs,
            output=result["output"],
            trace=result["trace"],
            started_at=started,
        )

    @app.get("/v1/workflows/{workflow_id}/runs", tags=["workflows"])
    def list_workflow_runs(workflow_id: str, user=Depends(get_current_user)) -> dict:
        _require_workflow(user, workflow_id)
        return {"runs": store.list_runs(workflow_id)}

    log.info(
        "chat_api_started",
        auth_mode=settings.auth_mode,
        models=len(registry.list_chat_models()),
        db=settings.db_engine,
    )
    return app


app = create_app()


def main() -> None:
    import uvicorn

    settings = ChatSettings()
    uvicorn.run("ai_agent_chat.app:app", host=settings.host, port=settings.port)


if __name__ == "__main__":
    main()
