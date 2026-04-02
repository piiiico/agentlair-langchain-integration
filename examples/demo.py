"""
AgentLair + LangChain — working demo.

This demo shows an agent that:
1. Has a persistent AgentLair identity (email address)
2. Uses AgentLair skills as LangChain tools (email, vault, audit trail)
3. Automatically logs every tool call to the AgentLair audit trail

Requirements:
    pip install langchain-agentlair langchain-openai

Environment variables:
    AGENTLAIR_API_KEY   — from https://agentlair.dev
    OPENAI_API_KEY      — from https://platform.openai.com

Usage:
    python examples/demo.py
"""

import os
import sys
import json

# Allow running from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_agentlair import AgentLairClient, agentlair_tools, AgentLairCallbackHandler

AGENTLAIR_API_KEY = os.environ.get("AGENTLAIR_API_KEY", "")
AGENT_EMAIL = os.environ.get("AGENTLAIR_EMAIL", "langchain-demo@agentlair.dev")


def demo_tools_directly():
    """
    Demo 1: Use AgentLair tools directly (no LLM required).
    Shows tool structure, agent identity, and audit trail.
    """
    print("=" * 60)
    print("Demo 1: AgentLair tools without LLM")
    print("=" * 60)

    client = AgentLairClient(api_key=AGENTLAIR_API_KEY)

    # Claim the agent's email address
    print(f"\n→ Claiming {AGENT_EMAIL}...")
    try:
        result = client.claim_email(AGENT_EMAIL)
        print(f"  ✓ {result.get('address')} claimed: {result.get('claimed')}")
    except Exception as e:
        print(f"  (already claimed or error: {e})")

    # Get all tools
    tools = agentlair_tools(client, AGENT_EMAIL)
    print(f"\n→ AgentLair tools available:")
    for tool in tools:
        print(f"  • {tool.name}: {tool.description[:60]}...")
        print(f"    identity: {tool.metadata}")

    # Store something in vault
    print(f"\n→ Storing a value in vault...")
    vault_tool = next(t for t in tools if t.name == "vault_store")
    result = vault_tool._run(key="demo-run", value="LangChain integration test 2026-04-02")
    print(f"  ✓ {result}")

    # Log an observation
    print(f"\n→ Logging observation to audit trail...")
    obs_tool = next(t for t in tools if t.name == "log_observation")
    result = obs_tool._run(
        topic="demo.started",
        content=json.dumps({"demo": "langchain-integration", "agent": AGENT_EMAIL}),
        display_name="Demo started",
    )
    print(f"  ✓ {result}")

    # Check audit trail
    print(f"\n→ Recent audit trail:")
    observations = client.get_observations(limit=5).get("observations", [])
    for obs in observations[:3]:
        print(f"  [{obs.get('topic')}] {obs.get('display_name', '')} — {obs.get('created_at', '')}")

    print()


def demo_with_openai():
    """
    Demo 2: Full AgentLair agent with OpenAI LLM.
    Requires OPENAI_API_KEY.
    """
    print("=" * 60)
    print("Demo 2: AgentLair agent with LLM")
    print("=" * 60)

    openai_key = os.environ.get("OPENAI_API_KEY", "")
    if not openai_key:
        print("  Skipping: OPENAI_API_KEY not set")
        return

    try:
        from langchain_openai import ChatOpenAI
        from langchain_agentlair import AgentLairAgent
    except ImportError as e:
        print(f"  Skipping: {e}")
        return

    llm = ChatOpenAI(model="gpt-4o-mini", api_key=openai_key)

    agent = AgentLairAgent(
        llm=llm,
        agent_email=AGENT_EMAIL,
        verbose=True,
    )

    print(f"\n→ Running agent as {AGENT_EMAIL}...")
    result = agent.run(
        "Store the value 'hello from langchain' under the vault key 'langchain-test', "
        "then log an observation with topic 'task.completed' noting what you did."
    )

    print(f"\n→ Agent output:\n{result}")

    print(f"\n→ Audit trail (last 5 entries):")
    trail = agent.get_audit_trail(limit=5)
    for entry in trail[:5]:
        print(f"  [{entry.get('topic')}] {entry.get('display_name', '')} — {entry.get('created_at', '')}")

    print()


def demo_callback_handler():
    """
    Demo 3: AgentLairCallbackHandler attached to an existing LangChain chain.
    Shows how to add audit trail to any existing LangChain setup.
    """
    print("=" * 60)
    print("Demo 3: AgentLairCallbackHandler on existing chain")
    print("=" * 60)

    client = AgentLairClient(api_key=AGENTLAIR_API_KEY)
    handler = AgentLairCallbackHandler(client=client, agent_email=AGENT_EMAIL)

    print(f"\n→ Callback handler created for {AGENT_EMAIL}")
    print(f"  Attach to any LangChain component with: callbacks=[handler]")
    print()
    print("  Example:")
    print("    from langchain_openai import ChatOpenAI")
    print("    llm = ChatOpenAI(callbacks=[handler])")
    print("    agent = AgentExecutor(agent=..., tools=tools, callbacks=[handler])")
    print()
    print("  Every tool call will be logged to:")
    print("    GET https://agentlair.dev/v1/observations")
    print()


if __name__ == "__main__":
    if not AGENTLAIR_API_KEY:
        print("Error: AGENTLAIR_API_KEY environment variable not set.")
        print("Get your key at https://agentlair.dev")
        sys.exit(1)

    demo_tools_directly()
    demo_callback_handler()
    demo_with_openai()

    print("✓ Demo complete. View your audit trail at https://agentlair.dev")
