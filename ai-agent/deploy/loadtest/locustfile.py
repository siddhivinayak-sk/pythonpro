"""Load/latency smoke test for the chat API (Locust).

Usage (against a running stack):
    pip install locust
    LOCUST_ADMIN_PASSWORD=... locust -f deploy/loadtest/locustfile.py --host http://localhost:8080

Logs in as the local admin, then repeatedly creates a conversation and sends a chat turn. Use it to
observe first-token latency, throughput, and error rates under concurrency (see nfr-governance §2 SLOs).
"""

from __future__ import annotations

import os

from locust import HttpUser, between, task


class ChatUser(HttpUser):
    wait_time = between(1, 3)

    def on_start(self) -> None:
        resp = self.client.post(
            "/v1/auth/login",
            json={
                "username": os.environ.get("LOCUST_ADMIN_USER", "admin"),
                "password": os.environ.get("LOCUST_ADMIN_PASSWORD", "admin"),
            },
        )
        token = resp.json().get("token") if resp.status_code == 200 else None
        self.client.headers.update({"Authorization": f"Bearer {token}"} if token else {})

    @task(3)
    def chat_turn(self) -> None:
        conv = self.client.post("/v1/conversations", json={}).json()
        self.client.post(
            f"/v1/conversations/{conv['id']}/chat",
            json={"text": "Give me a one-sentence summary of the leave policy."},
            name="/v1/conversations/[id]/chat",
        )

    @task(1)
    def list_models(self) -> None:
        self.client.get("/v1/models")
