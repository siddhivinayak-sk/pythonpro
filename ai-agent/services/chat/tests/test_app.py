"""End-to-end tests for the chat API (login → chat → switch model → resume → vision), fake model."""

from __future__ import annotations

from pathlib import Path

from ai_agent_chat.app import create_app
from ai_agent_chat.config import ChatSettings
from fastapi.testclient import TestClient


class _Chunk:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeModel:
    def invoke(self, messages: list):
        return _Chunk("hello from the assistant")

    def stream(self, messages: list):
        for piece in ["hello", " from", " the", " assistant"]:
            yield _Chunk(piece)


def _client(uploads_dir: str | None = None) -> TestClient:
    settings = ChatSettings(
        db_path=":memory:",
        admin_username="admin",
        admin_password="secret",
        log_json=False,
        uploads_dir=uploads_dir or "./data/chat/uploads",
    )
    app = create_app(settings, model_provider=lambda c, m, p: _FakeModel())
    return TestClient(app)


def _login(client: TestClient) -> dict:
    resp = client.post("/v1/auth/login", json={"username": "admin", "password": "secret"})
    assert resp.status_code == 200, resp.text
    client.headers.update({"Authorization": f"Bearer {resp.json()['token']}"})
    return resp.json()


def test_requires_auth() -> None:
    assert _client().get("/v1/conversations").status_code == 401


def test_login_and_me() -> None:
    client = _client()
    data = _login(client)
    assert data["user"]["role"] == "admin"
    assert client.get("/v1/auth/me").json()["role"] == "admin"


def test_models_listed_after_login() -> None:
    client = _client()
    _login(client)
    models = client.get("/v1/models").json()
    assert any(m["model_name"] == "llama3.1" for m in models)


def test_chat_flow_switch_model_and_resume() -> None:
    client = _client()
    _login(client)
    cid = client.post("/v1/conversations", json={}).json()["id"]

    r1 = client.post(
        f"/v1/conversations/{cid}/chat",
        json={"text": "hi", "connection_id": "ollama-local", "model_name": "llama3.1"},
    )
    assert r1.status_code == 200
    assert r1.json()["content"] == "hello from the assistant"
    assert r1.json()["model_name"] == "llama3.1"

    r2 = client.post(
        f"/v1/conversations/{cid}/chat",
        json={"text": "again", "connection_id": "openai-prod", "model_name": "gpt-4o"},
    )
    assert r2.json()["model_name"] == "gpt-4o"  # switched mid-conversation

    messages = client.get(f"/v1/conversations/{cid}/messages").json()["messages"]
    assert len(messages) == 4  # 2 user + 2 assistant, history preserved for resume
    assert any(c["id"] == cid for c in client.get("/v1/conversations").json()["conversations"])


def test_streaming_endpoint() -> None:
    client = _client()
    _login(client)
    cid = client.post("/v1/conversations", json={}).json()["id"]
    with client.stream("POST", f"/v1/conversations/{cid}/chat/stream", json={"text": "hi"}) as resp:
        body = "".join(resp.iter_text())
    assert "hello" in body
    assert "[DONE]" in body


def test_settings_precedence_via_api() -> None:
    client = _client()
    _login(client)
    client.put("/v1/settings", json={"data": {"temperature": 0.2, "theme": "dark"}})
    assert client.get("/v1/settings").json()["settings"]["temperature"] == 0.2


def test_vision_upload_and_chat(tmp_path: Path) -> None:
    client = _client(uploads_dir=str(tmp_path / "uploads"))
    _login(client)
    cid = client.post("/v1/conversations", json={}).json()["id"]

    up = client.post(
        "/v1/uploads", files={"file": ("x.png", b"\x89PNG\r\n\x1a\n fake", "image/png")}
    )
    assert up.status_code == 200
    url = up.json()["url"]
    assert url.startswith("/v1/uploads/")

    chat = client.post(
        f"/v1/conversations/{cid}/chat",
        json={"text": "what is in this image?", "images": ["data:image/png;base64,AAAA"]},
    )
    assert chat.status_code == 200
    assert client.get(url).status_code == 200  # uploaded file is retrievable by its owner


def test_logout_invalidates_session() -> None:
    client = _client()
    _login(client)
    assert client.post("/v1/auth/logout").status_code == 204
    assert client.get("/v1/auth/me").status_code == 401


# --- Phase 4: agentic chat (RAG + web citations) and workflows -------------------------------------
def _agentic_client() -> TestClient:
    from ai_agent_chat.agent import ModelTurn, ToolInvocation
    from ai_agent_chat.tools import FunctionTool, ToolResult

    settings = ChatSettings(
        db_path=":memory:", admin_username="admin", admin_password="secret", log_json=False
    )

    script = [
        ModelTurn(
            "",
            [
                ToolInvocation("1", "retrieve", {"query": "leave"}),
                ToolInvocation("2", "web_search", {"query": "leave"}),
            ],
        ),
        ModelTurn("Grounded answer.", []),
    ]

    class _FakeAgentModel:
        def __init__(self) -> None:
            self._i = 0

        def bind_tools(self, specs):
            return self

        def invoke(self, messages):
            turn = script[min(self._i, len(script) - 1)]
            self._i += 1
            return turn

    rag_tool = FunctionTool(
        "retrieve",
        "retrieve",
        {"type": "object", "properties": {"query": {"type": "string"}}},
        lambda query: ToolResult("passage", [{"path": "hr/leave.pdf", "source": "rag"}]),
    )
    web_tool = FunctionTool(
        "web_search",
        "web",
        {"type": "object", "properties": {"query": {"type": "string"}}},
        lambda query: ToolResult("web text", [{"url": "https://e/1", "source": "web"}]),
    )
    app = create_app(
        settings,
        agent_model_provider=lambda c, m, p: _FakeAgentModel(),
        tool_builder=lambda s, eff: [rag_tool, web_tool],
    )
    return TestClient(app)


def test_agentic_chat_cites_rag_and_web() -> None:
    client = _agentic_client()
    _login(client)
    cid = client.post("/v1/conversations", json={}).json()["id"]
    resp = client.post(f"/v1/conversations/{cid}/chat", json={"text": "what is the leave policy?"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["content"] == "Grounded answer."
    assert {c.get("source") for c in body["citations"]} == {"rag", "web"}
    assert set(body["tools_used"]) == {"retrieve", "web_search"}


def _workflow_client() -> TestClient:
    settings = ChatSettings(
        db_path=":memory:", admin_username="admin", admin_password="secret", log_json=False
    )

    class _Out:
        def __init__(self, content: str) -> None:
            self.content = content

    class _FakeWfModel:
        def invoke(self, messages):
            return _Out("workflow output: " + getattr(messages[-1], "content", ""))

    app = create_app(settings, workflow_model_provider=lambda c, m, p: _FakeWfModel())
    return TestClient(app)


def test_workflow_create_and_run() -> None:
    client = _workflow_client()
    _login(client)
    definition = {
        "steps": [{"id": "s1", "type": "llm", "prompt": "Summarize {{inputs.topic}}"}],
        "output": "{{s1}}",
    }
    wf = client.post("/v1/workflows", json={"name": "Summarizer", "definition": definition}).json()
    assert wf["name"] == "Summarizer"

    run = client.post(
        f"/v1/workflows/{wf['id']}/run", json={"inputs": {"topic": "leave policy"}}
    ).json()
    assert run["status"] == "completed"
    assert "workflow output" in run["output"]
    assert "leave policy" in run["output"]

    runs = client.get(f"/v1/workflows/{wf['id']}/runs").json()["runs"]
    assert len(runs) == 1


def test_metrics_and_security_headers() -> None:
    client = _client()
    assert client.get("/metrics").status_code == 200  # Prometheus endpoint wired
    assert client.get("/healthz").headers["x-content-type-options"] == "nosniff"


# --- context files + audio (brief gaps closed) ------------------------------------------------------
def test_context_upload_list_and_used_in_chat(tmp_path: Path) -> None:
    client = _client(uploads_dir=str(tmp_path))
    _login(client)
    cid = client.post("/v1/conversations", json={}).json()["id"]
    up = client.post(
        f"/v1/conversations/{cid}/context",
        files={"file": ("policy.txt", b"leave policy details", "text/plain")},
    )
    assert up.status_code == 200
    assert up.json()["chars"] > 0
    listed = client.get(f"/v1/conversations/{cid}/context").json()["context"]
    assert listed[0]["filename"] == "policy.txt"
    # chat still works with context injected
    assert (
        client.post(f"/v1/conversations/{cid}/chat", json={"text": "summarize"}).status_code == 200
    )


def test_audio_returns_501_when_unconfigured() -> None:
    client = _client()
    _login(client)
    assert client.post("/v1/audio/speech", json={"text": "hello"}).status_code == 501


def _audio_client() -> TestClient:
    class _FakeSpeech:
        def transcribe(self, audio: bytes, mime: str) -> str:
            return "transcribed text"

        def synthesize(self, text: str, voice: str):
            return (b"AUDIO", "audio/mpeg")

    settings = ChatSettings(
        db_path=":memory:",
        admin_username="admin",
        admin_password="secret",
        log_json=False,
        audio_enabled=True,
    )
    app = create_app(
        settings, model_provider=lambda c, m, p: _FakeModel(), speech_provider=_FakeSpeech()
    )
    return TestClient(app)


def test_audio_with_provider() -> None:
    client = _audio_client()
    _login(client)
    t = client.post("/v1/audio/transcribe", files={"file": ("a.webm", b"\x00\x01", "audio/webm")})
    assert t.status_code == 200
    assert t.json()["text"] == "transcribed text"
    s = client.post("/v1/audio/speech", json={"text": "hi", "voice": "x"})
    assert s.status_code == 200
    assert s.content == b"AUDIO"


# --- discovery endpoints + per-conversation settings ------------------------------------------------
def test_mcp_servers_endpoint_lists_configured() -> None:
    settings = ChatSettings(
        db_path=":memory:",
        admin_username="admin",
        admin_password="secret",
        log_json=False,
        mcp_servers=["http://mcp:8090"],
    )
    client = TestClient(create_app(settings, model_provider=lambda c, m, p: _FakeModel()))
    _login(client)
    assert client.get("/v1/mcp/servers").json()["servers"] == ["http://mcp:8090"]


def test_rag_collections_empty_when_unconfigured() -> None:
    client = _client()  # no rag_api_base_url
    _login(client)
    assert client.get("/v1/rag/collections").json()["collections"] == []


def test_rag_collections_proxies_rag_service(monkeypatch) -> None:
    from ai_agent_chat import rag_client

    monkeypatch.setattr(
        rag_client.RagClient, "list_collections", lambda self: ["handbook", "policies"]
    )
    settings = ChatSettings(
        db_path=":memory:",
        admin_username="admin",
        admin_password="secret",
        log_json=False,
        rag_api_base_url="http://rag:8081",
    )
    client = TestClient(create_app(settings, model_provider=lambda c, m, p: _FakeModel()))
    _login(client)
    assert client.get("/v1/rag/collections").json()["collections"] == ["handbook", "policies"]


def test_conversation_settings_get_put_and_ownership() -> None:
    client = _client()
    _login(client)
    cid = client.post("/v1/conversations", json={}).json()["id"]

    # defaults present before any override
    before = client.get(f"/v1/conversations/{cid}/settings").json()["settings"]
    assert "temperature" in before

    # per-conversation override applies
    put = client.put(f"/v1/conversations/{cid}/settings", json={"data": {"temperature": 0.1}})
    assert put.status_code == 200
    assert client.get(f"/v1/conversations/{cid}/settings").json()["settings"]["temperature"] == 0.1

    # unknown conversation -> 404 (ownership/existence guard)
    assert client.get("/v1/conversations/does-not-exist/settings").status_code == 404
