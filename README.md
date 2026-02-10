# Agent Runtime SDK

A minimal, opinionated SDK for building agentic loops with MCP (Model Context Protocol) tool integration.

## Features

- **Agent / Thread / Run architecture** - Clean separation of configuration, conversation, and execution
- **MCP integration** - Connect to MCP servers (stdio, SSE) for tool discovery and execution
- **Model-agnostic** - Uses LangChain core for provider flexibility (OpenAI in v0, more coming)
- **Persistence-ready** - In-memory storage with swappable store protocol for future DB integration

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

## Core Concepts

### Agent

Persistent configuration that defines an agent's behavior:
- Model and provider settings
- System prompt
- MCP servers available to the agent
- Execution parameters (max iterations, temperature)

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

## Future Features (Designed For, Not Yet Implemented)

- **Per-run MCP server overrides** - Enable/disable specific servers per run
- **Thread forking** - Copy a thread's messages to explore alternate paths
- **DB persistence** - Swap `InMemoryStore` for `PrismaStore` or `PostgresStore`
- **Streaming** - Real-time token streaming
- **Multiple providers** - Anthropic, others (just swap `ChatOpenAI` for `ChatAnthropic`)

## Environment Variables

- `OPENAI_API_KEY` - Required for OpenAI models (or pass `api_key` to agent)

## Project Structure

```
agent_runtime/
  __init__.py          # Public API exports
  config.py            # MCPServerConfig
  entities.py          # Agent, Thread, Run
  store.py             # Store protocol + InMemoryStore
  mcp_manager.py       # MCP connection lifecycle
  tool_adapter.py      # MCP → LangChain tool conversion
  runtime.py           # AgentRuntime (main orchestrator)
  models.py            # StepResult, RunResult

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
