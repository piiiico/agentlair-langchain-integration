"""
AgentLair API client — thin wrapper around the AgentLair REST API.
"""
from __future__ import annotations

import os
from typing import Any

import requests


class AgentLairClient:
    """Minimal AgentLair REST client used by LangChain tools."""

    BASE_URL = "https://agentlair.dev"

    def __init__(self, api_key: str | None = None, base_url: str | None = None):
        self.api_key = api_key or os.environ.get("AGENTLAIR_API_KEY", "")
        if not self.api_key:
            raise ValueError(
                "AgentLair API key required. "
                "Pass api_key= or set AGENTLAIR_API_KEY env var."
            )
        self.base_url = (base_url or os.environ.get("AGENTLAIR_BASE_URL", self.BASE_URL)).rstrip("/")

    @property
    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _get(self, path: str, params: dict | None = None) -> Any:
        resp = requests.get(f"{self.base_url}{path}", headers=self._headers, params=params, timeout=30)
        resp.raise_for_status()
        return resp.json()

    def _post(self, path: str, body: dict | None = None) -> Any:
        resp = requests.post(f"{self.base_url}{path}", headers=self._headers, json=body or {}, timeout=30)
        resp.raise_for_status()
        return resp.json()

    # ── Email ──────────────────────────────────────────────────────────────

    def claim_email(self, address: str) -> dict:
        return self._post("/v1/email/claim", {"address": address})

    def send_email(self, from_: str, to: str | list[str], subject: str, text: str) -> dict:
        if isinstance(to, str):
            to = [to]
        return self._post("/v1/email/send", {"from": from_, "to": to, "subject": subject, "text": text})

    def check_inbox(self, address: str, limit: int = 20) -> dict:
        return self._get("/v1/email/inbox", {"address": address, "limit": limit})

    # ── Vault ──────────────────────────────────────────────────────────────

    def vault_store(self, key: str, value: str) -> dict:
        resp = requests.put(
            f"{self.base_url}/v1/vault/{key}",
            headers=self._headers,
            json={"ciphertext": value},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()

    def vault_get(self, key: str) -> dict:
        return self._get(f"/v1/vault/{key}")

    def vault_list(self) -> dict:
        return self._get("/v1/vault")

    # ── Audit trail (observations) ─────────────────────────────────────────

    def log_observation(self, topic: str, content: dict, display_name: str = "") -> dict:
        import json as _json
        body: dict = {"topic": topic, "content": _json.dumps(content)}
        if display_name:
            body["display_name"] = display_name
        return self._post("/v1/observations", body)

    def get_observations(self, topic: str | None = None, limit: int = 50) -> dict:
        params: dict = {"limit": limit}
        if topic:
            params["topic"] = topic
        return self._get("/v1/observations", params)

    # ── Account ────────────────────────────────────────────────────────────

    def get_account(self) -> dict:
        return self._get("/v1/account/me")
