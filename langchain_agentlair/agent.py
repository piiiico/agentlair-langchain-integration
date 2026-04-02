"""
Pre-configured LangChain agent with AgentLair identity.

AgentLairAgent wraps LangChain's AgentExecutor and wires up:
  - AgentLair tools (email, vault, audit trail)
  - AgentLair audit trail callback (automatic logging of all tool calls)
  - Agent identity (email address injected into system prompt)

Usage::

    from langchain_agentlair import AgentLairAgent
    from langchain_openai import ChatOpenAI

    agent = AgentLairAgent.from_env(
        llm=ChatOpenAI(model="gpt-4o"),
        agent_email="mybot@agentlair.dev",
    )
    result = agent.run("Check my inbox and summarise any unread messages.")
    print(result)
"""
from __future__ import annotations

import os
from typing import Any, Optional

try:
    from langchain import hub  # type: ignore[attr-defined]
except ImportError:
    hub = None  # type: ignore[assignment]
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool

from .callbacks import AgentLairCallbackHandler
from .client import AgentLairClient
from .tools import agentlair_tools


SYSTEM_PROMPT_TEMPLATE = """\
You are {agent_email}, an autonomous AI agent.

Your AgentLair identity gives you:
- A persistent email address: {agent_email}
- An encrypted vault for cross-session state
- An immutable audit trail for all your actions

Always use log_observation to record important decisions or actions you take.
When you complete a task, write a final observation with topic "task.completed".
"""


class AgentLairAgent:
    """
    A LangChain agent pre-configured with AgentLair identity and skills.

    Parameters
    ----------
    llm:
        Any LangChain chat model (ChatOpenAI, ChatAnthropic, etc.)
    agent_email:
        The @agentlair.dev address this agent operates as.
    client:
        AgentLairClient instance. Defaults to AGENTLAIR_API_KEY env var.
    extra_tools:
        Additional LangChain tools to include alongside AgentLair skills.
    verbose:
        Pass-through to AgentExecutor.
    """

    def __init__(
        self,
        llm: BaseChatModel,
        agent_email: str,
        client: Optional[AgentLairClient] = None,
        extra_tools: Optional[list[BaseTool]] = None,
        verbose: bool = True,
    ):
        self.agent_email = agent_email
        self.client = client or AgentLairClient()

        # Claim the address (idempotent — safe to call every time)
        try:
            self.client.claim_email(agent_email)
        except Exception:
            pass  # Already claimed or rate-limited — continue

        # Build tools
        tools = agentlair_tools(self.client, agent_email)
        if extra_tools:
            tools.extend(extra_tools)

        # Audit trail callback
        self.callback = AgentLairCallbackHandler(self.client, agent_email)

        # Build agent with ReAct prompt
        try:
            prompt = hub.pull("hwchase17/react")
        except Exception:
            # Offline fallback — minimal ReAct template
            from langchain_core.prompts import PromptTemplate
            prompt = PromptTemplate.from_template(
                "You are a helpful assistant with access to tools.\n\n"
                "Tools: {tools}\nTool names: {tool_names}\n\n"
                "Question: {input}\n{agent_scratchpad}"
            )

        react_agent = create_react_agent(llm, tools, prompt)
        self.executor = AgentExecutor(
            agent=react_agent,
            tools=tools,
            callbacks=[self.callback],
            verbose=verbose,
            handle_parsing_errors=True,
        )

        self._identity_context = SYSTEM_PROMPT_TEMPLATE.format(agent_email=agent_email)

    @classmethod
    def from_env(
        cls,
        llm: BaseChatModel,
        agent_email: Optional[str] = None,
        **kwargs: Any,
    ) -> "AgentLairAgent":
        """
        Create an AgentLairAgent using environment variables.

        Environment variables:
            AGENTLAIR_API_KEY   — required
            AGENTLAIR_EMAIL     — agent's email address (or pass agent_email=)
        """
        email = agent_email or os.environ.get("AGENTLAIR_EMAIL", "")
        if not email:
            raise ValueError(
                "Agent email address required. "
                "Pass agent_email= or set AGENTLAIR_EMAIL env var."
            )
        return cls(llm=llm, agent_email=email, **kwargs)

    def run(self, task: str, **kwargs: Any) -> str:
        """Run the agent on a task and return the final output."""
        full_input = f"{self._identity_context}\n\nTask: {task}"
        result = self.executor.invoke({"input": full_input}, **kwargs)
        return result.get("output", "")

    def get_audit_trail(self, topic: Optional[str] = None, limit: int = 50) -> list[dict]:
        """Fetch this agent's audit trail observations from AgentLair."""
        return self.client.get_observations(topic=topic, limit=limit).get("observations", [])
