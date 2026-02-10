# Architecture Overview

## Core Design Principle

**Thread is independent** - A thread doesn't "belong" to an agent. Instead, when you create a Run, you specify both `thread_id` and `agent_id`. This connects the agent configuration with the conversation history for that execution.

## Data Flow

```
┌─────────────────┐         ┌─────────────────┐
│     Agent       │         │     Thread      │
│  (config)       │         │   (messages)    │
│                 │         │                 │
│ - model{}       │         │ - messages[]    │
│ - system_prompt │         │ - metadata      │
│ - mcp_servers   │         │                 │
│ - max_iterations│         │                 │
└─────────────────┘         └─────────────────┘
        │                           │
        │                           │
        └────────────┬──────────────┘
                     │
                     ▼
              ┌─────────────┐
              │     Run     │
              │ (execution) │
              │             │
              │ - thread_id │
              │ - agent_id  │
              │ - status    │
              │ - iterations│
              └─────────────┘
```

## Key Benefits of This Design

1. **Try different agents on the same conversation**
   ```python
   # Same thread, different agents
   runtime.run(thread_id=thread.id, agent_id=gpt4_agent.id, user_message="...")
   runtime.run(thread_id=thread.id, agent_id=claude_agent.id, user_message="...")
   ```

2. **Thread forking is trivial** (future)
   ```python
   # Copy messages to a new thread
   new_thread = Thread(messages=old_thread.messages.copy())
   ```

3. **Clear execution history**
   ```python
   # See all runs on a thread
   runs = store.list_runs(thread_id=thread.id)
   # Each run shows which agent was used, iterations, stop reason
   ```

## Class Responsibilities

### `AgentRuntime`
- CRUD for agents and threads
- Orchestrates the agentic loop
- Connects agents, threads, and MCP servers

### `MCPManager`
- Connects to MCP servers (stdio, SSE)
- Discovers tools from servers
- Routes tool calls to the correct server
- Uses the official `mcp` Python client

### `ToolAdapter`
- Converts MCP tool schemas to LangChain format
- MCP tools → OpenAI function calling format → LangChain bind_tools()

### `Store`
- Persistence abstraction (protocol)
- `InMemoryStore` for v0
- Future: `PrismaStore`, `PostgresStore`

## The Agentic Loop

```python
# Simplified pseudocode
def run(thread_id, agent_id, user_message):
    thread = store.get_thread(thread_id)
    agent = store.get_agent(agent_id)
    run = Run(thread_id=thread.id, agent_id=agent.id, status="running")
    
    thread.messages.append(HumanMessage(content=user_message))
    
    # Connect to MCP servers, discover tools
    mcp_manager = MCPManager(agent.mcp_servers)
    await mcp_manager.connect_all()
    tools = mcp_manager.get_tools()
    
    # Create model with tools
    model = ChatOpenAI(model=agent.model).bind_tools(tools)
    
    for i in range(agent.max_iterations):
        response = model.invoke([SystemMessage(agent.system_prompt)] + thread.messages)
        thread.messages.append(response)
        
        if not response.tool_calls:
            break  # Done
        
        # Execute tools
        for tool_call in response.tool_calls:
            result = await mcp_manager.call_tool(tool_call["name"], tool_call["args"])
            thread.messages.append(ToolMessage(content=result, tool_call_id=tool_call["id"]))
    
    store.save_thread(thread)
    store.save_run(run)
    await mcp_manager.disconnect_all()
```

## Message Flow in a Conversation

```
User: "What's the weather in SF?"
  ↓
[HumanMessage: "What's the weather in SF?"]
  ↓
Model thinks... requests tool
  ↓
[AIMessage: tool_calls=[{name: "get_weather", args: {location: "SF"}}]]
  ↓
Execute tool via MCP
  ↓
[ToolMessage: "Sunny, 72°F"]
  ↓
Model synthesizes response
  ↓
[AIMessage: "It's sunny and 72°F in San Francisco!"]
  ↓
Return to user
```

All these messages are stored in the thread. When you resume with "What about NYC?", the full history is sent to the model.

## Persistence Strategy

Every entity has:
- `id: str` (UUID)
- `created_at: datetime`
- `updated_at: datetime` (Agent, Thread)
- `to_dict()` / `from_dict()` methods

JSON-blob fields (for example, `Agent.model`) are represented in code with
explicit typed/versioned schemas (`ModelConfigV0`) before being persisted.

This makes DB migration straightforward:
```python
# In-memory (v0)
store = InMemoryStore()

# Future: Postgres
store = PrismaStore(database_url="postgresql://...")
# Runtime code doesn't change!
```

## MCP Server Configuration

```python
MCPServerConfig(
    name="weather",           # Identifier
    transport="stdio",        # or "sse"
    command="uvx",            # For stdio
    args=["mcp-server-weather"],
    env={"API_KEY": "..."},   # Environment variables
    enabled=True,             # v0: always True
)
```

Multiple servers:
```python
agent = runtime.create_agent(
    mcp_servers=[weather_config, filesystem_config, database_config]
)
# All tools from all servers are available to the agent
```

## Extension Points (Future)

1. **Per-run server overrides**
   ```python
   # Not implemented yet, but designed for:
   runtime.run(..., enabled_servers=["weather", "filesystem"])
   ```

2. **Streaming**
   ```python
   # Change from model.invoke() to model.astream()
   async for chunk in model.astream(messages):
       yield chunk
   ```

3. **Multiple providers**
   ```python
   # Just swap the model class
   from langchain_anthropic import ChatAnthropic
   model = ChatAnthropic(model="claude-sonnet-4")
   ```

4. **Custom tools (non-MCP)**
   ```python
   # Add to the tools list alongside MCP tools
   all_tools = mcp_tools + custom_tools
   model = model.bind_tools(all_tools)
   ```
