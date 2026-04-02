"""
langchain-agentlair — AgentLair identity, skills, and audit trail for LangChain agents.

Quick start::

    from langchain_agentlair import AgentLairAgent, agentlair_tools, AgentLairCallbackHandler
    from langchain_openai import ChatOpenAI

    agent = AgentLairAgent.from_env(
        llm=ChatOpenAI(model="gpt-4o"),
        agent_email="mybot@agentlair.dev",
    )
    result = agent.run("Send a hello email to test@example.com")

See https://github.com/piiiico/agentlair-langchain for full docs.
"""

from .agent import AgentLairAgent
from .callbacks import AgentLairCallbackHandler
from .client import AgentLairClient
from .tools import (
    CheckInboxTool,
    LogObservationTool,
    SendEmailTool,
    VaultGetTool,
    VaultStoreTool,
    agentlair_tools,
)

__all__ = [
    "AgentLairAgent",
    "AgentLairCallbackHandler",
    "AgentLairClient",
    "agentlair_tools",
    "SendEmailTool",
    "CheckInboxTool",
    "VaultStoreTool",
    "VaultGetTool",
    "LogObservationTool",
]

__version__ = "0.1.0"
