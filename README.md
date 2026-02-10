# Agent Runtime SDK

A minimal, opinionated SDK for building agentic loops with MCP (Model Context Protocol) tool integration.

## Features

- **Agent / Thread / Run architecture** - Clean separation of configuration, conversation, and execution
- **MCP integration** - Connect to MCP servers (stdio, SSE) for tool discovery and execution
- **Model-agnostic** - Uses LangChain core for provider flexibility (OpenAI in v0, more coming)
- **Versioned model config** - Agent model/provider/params are stored as one typed, versioned JSON blob
- **Persistence-ready** - In-memory storage plus SQLAlchemy-backed persistence (`SQLAlchemyStore`)
- **Server-ready** - Optional FastAPI API layer for multi-consumer/runtime-as-a-service usage
- **Streaming-ready runs** - Persistent `RunEvent` log with replay + SSE streaming

## Installation

```bash
# Install dependencies
pip install -e .

# Or with uv
uv pip install -e .
```

## Quick Start

```python
from agent_runtime import AgentRuntime, InMemoryStore, MCPServerConfig

# Initialize runtime
store = InMemoryStore()
runtime = AgentRuntime(store=store)

# Create an agent
agent = runtime.create_agent(
    name="weather-bot",
    model="gpt-4o",
    system_prompt="You are a helpful weather assistant.",
    mcp_servers=[
        MCPServerConfig(
            name="weather",
            transport="stdio",
            command="uvx",
            args=["mcp-server-weather"],
        ),
    ],
)

# Create a thread (conversation)
thread = runtime.create_thread()

# Run the agent
result = runtime.run(
    thread_id=thread.id,
    agent_id=agent.id,
    user_message="What's the weather in San Francisco?",
)

print(result.output)
```

See `examples/basic_agent.py` for a complete working example.

## Run as an API Server (FastAPI)

You can run the runtime as an HTTP service:

```bash
agent-runtime-server
```

Or:

```bash
python -m agent_runtime.server
```

Then use endpoints like:

- `POST /agents`
- `POST /threads`
- `POST /threads/{thread_id}/fork`
- `POST /threads/{thread_id}/rerun`
- `POST /runs` (queue run for background worker)
- `POST /runs/execute` (sync execute compatibility)
- `GET /runs/{run_id}/events` (replay)
- `GET /runs/{run_id}/stream` (SSE)
- `POST /runs/{run_id}/cancel`
- `POST /runs/{run_id}/retry`
- `GET /threads/{thread_id}/runs`

Interactive API docs are available at `/docs`.

## Core Concepts

### Agent

Persistent configuration that defines an agent's behavior:
- Versioned model/provider settings (`ModelConfigV0`)
- System prompt
- MCP servers available to the agent
- Execution parameters (max iterations; model params live under `model.config`)

Create once, reuse across many threads and runs.

### Thread

A conversation - just a container for messages. Independent of any agent.

Key insight: The same thread can be used with different agents. This enables experimentation ("try this conversation with GPT-4 vs Claude").

### Run

One execution of the agentic loop. Records:
- Which thread and agent were used
- Execution status and timing
- Number of iterations
- Stop reason (end_turn, max_iterations, error)

A thread can have many runs (that's a multi-turn conversation).

## Architecture

```
Agent (config) + Thread (messages) → Run (execution)
                ↓
         AgentRuntime
                ↓
         MCP Servers → Tools
```

## MCP Integration

The SDK uses the official `mcp` Python client to connect to MCP servers:

- **stdio transport**: Launch local MCP servers (e.g., `uvx mcp-server-weather`)
- **SSE transport**: Connect to remote HTTP-based MCP servers

Tools are automatically discovered from connected servers and made available to the agent.

## Example: Multiple MCP Servers

```python
agent = runtime.create_agent(
    name="multi-tool-agent",
    model="gpt-4o",
    mcp_servers=[
        MCPServerConfig(
            name="weather",
            transport="stdio",
            command="uvx",
            args=["mcp-server-weather"],
        ),
        MCPServerConfig(
            name="filesystem",
            transport="stdio",
            command="uvx",
            args=["mcp-server-filesystem"],
            env={"ALLOWED_PATHS": "/tmp"},
        ),
    ],
)
```

## Resuming Conversations

```python
# First interaction
result1 = runtime.run(thread_id=thread.id, agent_id=agent.id, 
                      user_message="What's the weather in SF?")

# Resume the same thread
result2 = runtime.run(thread_id=thread.id, agent_id=agent.id,
                      user_message="What about NYC?")

# The agent has full context from previous messages
```

## Run Streaming and Reliability (v0.2)

The API server now executes runs through an in-process `RunManager`:

- `POST /runs` creates a queued run record and returns immediately
- one background worker executes queued runs
- each lifecycle step emits a persisted `RunEvent`
- clients can replay history and then live-stream via SSE without losing order

This gives a practical reliability baseline without external queue infrastructure.

## Future Features (Designed For, Not Yet Implemented)

- **Per-run MCP server overrides** - Enable/disable specific servers per run
- **Thread forking** - Copy a thread's messages to explore alternate paths
- **More DB backends** - `SQLAlchemyStore` is included; swap in additional stores as needed
- **Token-level streaming** - Current events are step/tool lifecycle level
- **Multiple providers** - Anthropic, others (just swap `ChatOpenAI` for `ChatAnthropic`)

## Environment Variables

- `OPENAI_API_KEY` - Required for OpenAI models (or pass `api_key` to agent)
- `AGENT_RUNTIME_STORE` - `in_memory` (default) or `sqlalchemy`
- `AGENT_RUNTIME_DATABASE_URL` - SQLAlchemy URL when using DB store
- `AGENT_RUNTIME_HOST` - API host (default `0.0.0.0`)
- `AGENT_RUNTIME_PORT` - API port (default `8000`)
- `AGENT_RUNTIME_RELOAD` - `true|false` for uvicorn reload

## Project Structure

```
agent_runtime/
  __init__.py          # Public API exports
  core/                # Runtime domain (types, config, entities, runtime)
  stores/              # Store protocol + store implementations
  mcp/                 # MCP integration modules
  server/              # FastAPI app, schemas, and server launcher

  # Backward-compatible module shims (legacy imports)
  config.py
  entities.py
  runtime.py
  store.py
  sqlalchemy_store.py
  mcp_manager.py
  tool_adapter.py
  models.py
  types.py

examples/
  basic_agent.py       # Usage example
```

## Development

The SDK is designed to be simple and extensible:

1. **Entities are serializable** - All core objects have `to_dict()` / `from_dict()` for persistence
2. **Store is a protocol** - Swap implementations without changing runtime code
3. **MCP is the tool standard** - Tools come from MCP servers, not custom code
4. **LangChain core for models** - Use any LangChain-supported provider

## License

MIT
