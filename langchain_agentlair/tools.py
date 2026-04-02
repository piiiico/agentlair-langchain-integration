"""
AgentLair skills as LangChain tools.

Each tool carries the agent's AgentLair identity (email address) in its
metadata so the LangChain agent always operates as a named, addressable entity.
"""
from __future__ import annotations

import json
from typing import Optional, Type

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from .client import AgentLairClient


# ── Input schemas ──────────────────────────────────────────────────────────────


class SendEmailInput(BaseModel):
    to: str = Field(description="Recipient email address")
    subject: str = Field(description="Email subject line")
    body: str = Field(description="Plain-text email body")


class CheckInboxInput(BaseModel):
    limit: int = Field(default=10, description="Max number of messages to return (1-50)")


class VaultStoreInput(BaseModel):
    key: str = Field(description="Vault key name (alphanumeric, hyphens OK)")
    value: str = Field(description="Value to store (treated as opaque text)")


class VaultGetInput(BaseModel):
    key: str = Field(description="Vault key name to retrieve")


class LogObservationInput(BaseModel):
    topic: str = Field(description="Observation topic, e.g. 'task.completed' or 'decision.made'")
    content: str = Field(description="JSON-serialisable content to log as the observation body")
    display_name: Optional[str] = Field(default="", description="Human-readable label for this entry")


# ── Tools ──────────────────────────────────────────────────────────────────────


class SendEmailTool(BaseTool):
    """Send an email from the agent's AgentLair address."""

    name: str = "send_email"
    description: str = (
        "Send an email from the agent's AgentLair email address. "
        "Use for outreach, notifications, or any communication that requires a real email."
    )
    args_schema: Type[BaseModel] = SendEmailInput

    client: AgentLairClient
    agent_email: str

    class Config:
        arbitrary_types_allowed = True

    def _get_identity(self) -> dict:
        return {"agent_email": self.agent_email, "provider": "agentlair"}

    def _run(
        self,
        to: str,
        subject: str,
        body: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        result = self.client.send_email(
            from_=self.agent_email,
            to=to,
            subject=subject,
            text=body,
        )
        return json.dumps(result)


class CheckInboxTool(BaseTool):
    """Check the agent's AgentLair inbox for new messages."""

    name: str = "check_inbox"
    description: str = (
        "Check the agent's AgentLair email inbox. "
        "Returns recent messages including sender, subject, and body preview."
    )
    args_schema: Type[BaseModel] = CheckInboxInput

    client: AgentLairClient
    agent_email: str

    class Config:
        arbitrary_types_allowed = True

    def _get_identity(self) -> dict:
        return {"agent_email": self.agent_email, "provider": "agentlair"}

    def _run(
        self,
        limit: int = 10,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        result = self.client.check_inbox(address=self.agent_email, limit=limit)
        messages = result.get("messages", [])
        if not messages:
            return "Inbox is empty."
        lines = []
        for msg in messages:
            lines.append(
                f"FROM: {msg.get('from', '?')} | SUBJECT: {msg.get('subject', '(no subject)')} | "
                f"RECEIVED: {msg.get('received_at', '?')}\n  {msg.get('body_preview', '')}"
            )
        return "\n\n".join(lines)


class VaultStoreTool(BaseTool):
    """Store a value in the agent's encrypted AgentLair vault."""

    name: str = "vault_store"
    description: str = (
        "Store a value in the agent's encrypted vault. "
        "Use for persisting credentials, results, or state across sessions."
    )
    args_schema: Type[BaseModel] = VaultStoreInput

    client: AgentLairClient
    agent_email: str

    class Config:
        arbitrary_types_allowed = True

    def _get_identity(self) -> dict:
        return {"agent_email": self.agent_email, "provider": "agentlair"}

    def _run(
        self,
        key: str,
        value: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        result = self.client.vault_store(key=key, value=value)
        return f"Stored '{key}' in vault. Version: {result.get('version', '?')}"


class VaultGetTool(BaseTool):
    """Retrieve a value from the agent's encrypted AgentLair vault."""

    name: str = "vault_get"
    description: str = (
        "Retrieve a value from the agent's encrypted vault by key name."
    )
    args_schema: Type[BaseModel] = VaultGetInput

    client: AgentLairClient
    agent_email: str

    class Config:
        arbitrary_types_allowed = True

    def _get_identity(self) -> dict:
        return {"agent_email": self.agent_email, "provider": "agentlair"}

    def _run(
        self,
        key: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        result = self.client.vault_get(key=key)
        return result.get("ciphertext", "(empty)")


class LogObservationTool(BaseTool):
    """Write an entry to the AgentLair audit trail."""

    name: str = "log_observation"
    description: str = (
        "Write an observation to the AgentLair audit trail. "
        "Use to record decisions, actions taken, or any auditable event. "
        "The audit trail is immutable and queryable."
    )
    args_schema: Type[BaseModel] = LogObservationInput

    client: AgentLairClient
    agent_email: str

    class Config:
        arbitrary_types_allowed = True

    def _get_identity(self) -> dict:
        return {"agent_email": self.agent_email, "provider": "agentlair"}

    def _run(
        self,
        topic: str,
        content: str,
        display_name: str = "",
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        try:
            content_dict = json.loads(content)
        except json.JSONDecodeError:
            content_dict = {"raw": content}

        result = self.client.log_observation(
            topic=topic,
            content=content_dict,
            display_name=display_name,
        )
        return f"Logged observation '{topic}' (id: {result.get('id', '?')})"


# ── Factory ────────────────────────────────────────────────────────────────────


def agentlair_tools(client: AgentLairClient, agent_email: str) -> list[BaseTool]:
    """Return all AgentLair tools pre-configured for a specific agent identity."""
    identity_meta = {"agent_email": agent_email, "provider": "agentlair"}
    kwargs = {"client": client, "agent_email": agent_email, "metadata": identity_meta}
    return [
        SendEmailTool(**kwargs),
        CheckInboxTool(**kwargs),
        VaultStoreTool(**kwargs),
        VaultGetTool(**kwargs),
        LogObservationTool(**kwargs),
    ]
