"""
AgentLair audit trail callback handler for LangChain.

Every tool invocation is automatically logged as an observation on the
AgentLair audit trail — giving you a persistent, queryable record of what
your agent did and when.
"""
from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional, Union
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

from .client import AgentLairClient


class AgentLairCallbackHandler(BaseCallbackHandler):
    """
    LangChain callback handler that writes every tool call to AgentLair's
    audit trail via POST /v1/observations.

    Usage::

        handler = AgentLairCallbackHandler(client=client, agent_email="bot@agentlair.dev")
        llm = ChatOpenAI(callbacks=[handler])
        agent = AgentExecutor(agent=..., tools=tools, callbacks=[handler])
    """

    def __init__(self, client: AgentLairClient, agent_email: str):
        super().__init__()
        self.client = client
        self.agent_email = agent_email
        self._tool_start_times: dict[str, float] = {}

    # ── Tool lifecycle ─────────────────────────────────────────────────────

    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        tool_name = serialized.get("name", "unknown_tool")
        self._tool_start_times[str(run_id)] = time.time()
        try:
            self.client.log_observation(
                topic="tool.start",
                content={
                    "agent_email": self.agent_email,
                    "tool": tool_name,
                    "input": input_str,
                    "run_id": str(run_id),
                },
                display_name=f"{tool_name} → start",
            )
        except Exception:
            pass  # Audit failure must never break the agent

    def on_tool_end(
        self,
        output: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        elapsed = time.time() - self._tool_start_times.pop(str(run_id), time.time())
        try:
            self.client.log_observation(
                topic="tool.end",
                content={
                    "agent_email": self.agent_email,
                    "output_preview": str(output)[:500],
                    "elapsed_ms": round(elapsed * 1000),
                    "run_id": str(run_id),
                },
                display_name="tool → end",
            )
        except Exception:
            pass

    def on_tool_error(
        self,
        error: Union[Exception, KeyboardInterrupt],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        elapsed = time.time() - self._tool_start_times.pop(str(run_id), time.time())
        try:
            self.client.log_observation(
                topic="tool.error",
                content={
                    "agent_email": self.agent_email,
                    "error": str(error),
                    "elapsed_ms": round(elapsed * 1000),
                    "run_id": str(run_id),
                },
                display_name="tool → ERROR",
            )
        except Exception:
            pass

    # ── Agent lifecycle ────────────────────────────────────────────────────

    def on_agent_action(self, action: Any, *, run_id: UUID, **kwargs: Any) -> None:
        try:
            self.client.log_observation(
                topic="agent.action",
                content={
                    "agent_email": self.agent_email,
                    "tool": action.tool,
                    "tool_input": str(action.tool_input)[:500],
                    "run_id": str(run_id),
                },
                display_name=f"→ {action.tool}",
            )
        except Exception:
            pass

    def on_agent_finish(self, finish: Any, *, run_id: UUID, **kwargs: Any) -> None:
        try:
            output = finish.return_values.get("output", "")
            self.client.log_observation(
                topic="agent.finish",
                content={
                    "agent_email": self.agent_email,
                    "output_preview": str(output)[:500],
                    "run_id": str(run_id),
                },
                display_name="agent → finish",
            )
        except Exception:
            pass

    def on_chain_error(
        self,
        error: Union[Exception, KeyboardInterrupt],
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        try:
            self.client.log_observation(
                topic="agent.error",
                content={
                    "agent_email": self.agent_email,
                    "error": str(error),
                    "run_id": str(run_id),
                },
                display_name="agent → ERROR",
            )
        except Exception:
            pass
