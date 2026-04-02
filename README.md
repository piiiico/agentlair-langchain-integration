# langchain-agentlair

AgentLair identity, skills, and audit trail for LangChain agents.

Give your LangChain agent a persistent identity — a real email address, encrypted vault, and an immutable audit trail — in three lines of code.

```python
from langchain_agentlair import AgentLairAgent
from langchain_openai import ChatOpenAI

agent = AgentLairAgent.from_env(
    llm=ChatOpenAI(model="gpt-4o"),
    agent_email="mybot@agentlair.dev",
)
result = agent.run("Check my inbox and summarise any unread messages.")
```

## What this gives your agent

| Capability | What it means |
|------------|---------------|
| **Persistent identity** | A real `@agentlair.dev` email address that survives session restarts |
| **Email skills** | Send and receive real email — no SMTP configuration |
| **Encrypted vault** | Store secrets and state that persist across sessions |
| **Audit trail** | Every tool call logged to an immutable, queryable observation log |
| **Identity in metadata** | Every tool carries the agent's email in `.metadata` for downstream tracing |

## Why this matters

LangChain agents are stateless by default. Between sessions, they forget everything: credentials, context, decisions made.

AgentLair provides the stateful identity layer:
- **Cross-session**: the agent's email address, vault, and audit trail persist forever
- **Accountable**: every action is logged with a timestamp and agent identifier
- **Addressable**: external systems can reach your agent at its email address

## Installation

```bash
pip install langchain-agentlair

# With OpenAI
pip install langchain-agentlair langchain-openai

# With Anthropic
pip install langchain-agentlair langchain-anthropic
```

## Setup

Get an AgentLair API key at [agentlair.dev](https://agentlair.dev).

```bash
export AGENTLAIR_API_KEY=al_live_...
export AGENTLAIR_EMAIL=mybot@agentlair.dev  # your agent's address
```

## Usage

### Option A: Pre-built agent (recommended)

```python
from langchain_agentlair import AgentLairAgent
from langchain_openai import ChatOpenAI

agent = AgentLairAgent(
    llm=ChatOpenAI(model="gpt-4o"),
    agent_email="mybot@agentlair.dev",
)

result = agent.run("Store 'hello world' in my vault under key 'greeting'")
print(result)

# Fetch this agent's audit trail
trail = agent.get_audit_trail(limit=10)
```

### Option B: Use AgentLair tools in your own agent

```python
from langchain_agentlair import AgentLairClient, agentlair_tools
from langchain.agents import AgentExecutor, create_react_agent
from langchain import hub

client = AgentLairClient()  # reads AGENTLAIR_API_KEY
tools = agentlair_tools(client, agent_email="mybot@agentlair.dev")

# tools is a list of LangChain BaseTool objects:
# - send_email
# - check_inbox
# - vault_store
# - vault_get
# - log_observation

prompt = hub.pull("hwchase17/react")
agent = create_react_agent(llm, tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools)
```

### Option C: Add audit trail to an existing agent

```python
from langchain_agentlair import AgentLairClient, AgentLairCallbackHandler

client = AgentLairClient()
handler = AgentLairCallbackHandler(client=client, agent_email="mybot@agentlair.dev")

# Attach to any LangChain component
llm = ChatOpenAI(callbacks=[handler])
executor = AgentExecutor(agent=agent, tools=tools, callbacks=[handler])
```

Every tool call now logs to AgentLair:
- `tool.start` — before tool runs (records input)
- `tool.end` — after tool succeeds (records output preview + elapsed ms)
- `tool.error` — if tool raises
- `agent.action` — when agent selects a tool
- `agent.finish` — when agent returns final answer
- `agent.error` — if agent chain fails

## Available tools

| Tool | Description |
|------|-------------|
| `send_email` | Send email from the agent's `@agentlair.dev` address |
| `check_inbox` | Read the agent's inbox |
| `vault_store` | Store a value in the agent's encrypted vault |
| `vault_get` | Retrieve a value from the vault |
| `log_observation` | Write an entry to the audit trail |

All tools carry `tool.metadata["agent_email"]` — the agent's persistent identity.

## Agent identity in tool metadata

```python
tools = agentlair_tools(client, "mybot@agentlair.dev")
for tool in tools:
    print(tool.name, tool.metadata)
# send_email {'agent_email': 'mybot@agentlair.dev', 'provider': 'agentlair'}
# check_inbox {'agent_email': 'mybot@agentlair.dev', 'provider': 'agentlair'}
# ...
```

## Running the demo

```bash
git clone https://github.com/piiiico/agentlair-langchain
cd agentlair-langchain
pip install -e ".[dev]"

export AGENTLAIR_API_KEY=al_live_...
python examples/demo.py
```

## Relationship to OpenBox / other governance tools

AgentLair is an **identity-scoped** layer — it answers "who is this agent, what has it done, and what is it authorized to do?" across sessions.

Tools like [OpenBox](https://openbox.dev) provide **session-scoped** governance — they intercept actions within a single execution context. These are additive:

```
OpenBox:   "This action has risk score 0.73"
AgentLair: "And this agent is mybot@agentlair.dev (trusted since 2026-01-01,
            authorized by human@company.com for this task)"
```

AgentLair can act as the identity resolver for OpenBox policy decisions.

## License

MIT — see [LICENSE](LICENSE).

---

Built for [AgentLair](https://agentlair.dev) — identity infrastructure for autonomous agents.
