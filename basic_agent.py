"""
Basic usage example of the Agent Runtime SDK.

This example demonstrates:
1. Creating an agent with MCP server configuration
2. Creating a thread (conversation)
3. Running the agent on the thread
4. Resuming the conversation with another message
"""

import os

import sys
import os

from agent_runtime import AgentRuntime
from agent_runtime import InMemoryStore
from agent_runtime import MCPServerConfig
from dotenv import load_dotenv

load_dotenv()



def main():
    # Initialize the runtime with an in-memory store
    store = InMemoryStore()
    runtime = AgentRuntime(store=store)

    # Create an agent with a weather MCP server
    # Note: Make sure you have the mcp-server-weather installed:
    #   pip install mcp-server-weather
    # Or use `uvx mcp-server-weather` without installation
    agent = runtime.create_agent(
        name="weather-assistant",
        model="gpt-4o",
        system_prompt="You are a helpful timezone assistant. Use the available tools to answer time and timezone questions.",
        mcp_servers=[
            MCPServerConfig(
                name="time",
                transport="stdio",
                command="uvx",
                args=["mcp-server-time"],
            ),
        ],
        temperature=0.0,
    )

    print(f"✓ Created agent: {agent.name} (ID: {agent.id})")

    # Create a thread (conversation)
    thread = runtime.create_thread(
        metadata={"user_id": "demo_user", "session": "example_1"}
    )

    print(f"✓ Created thread (ID: {thread.id})")

    # Run the agent - first message
    print("\n--- First turn ---")
    result1 = runtime.run(
        thread_id=thread.id,
        agent_id=agent.id,
        user_message="What's the time in Tokyo?",
    )

    print(f"Agent: {result1.output}")
    print(f"Run ID: {result1.run_id}")
    print(f"Iterations: {result1.iterations}")
    print(f"Stop reason: {result1.stop_reason}")

    # Resume the conversation
    print("\n--- Second turn ---")
    result2 = runtime.run(
        thread_id=thread.id,
        agent_id=agent.id,
        user_message="What's the time in London?",
    )

    print(f"Agent: {result2.output}")
    print(f"Run ID: {result2.run_id}")
    print(f"Iterations: {result2.iterations}")

    # Inspect the thread history
    print("\n--- Thread history ---")
    updated_thread = runtime.get_thread(thread.id)
    if updated_thread:
        print(f"Total messages in thread: {len(updated_thread.messages)}")
        for i, msg in enumerate(updated_thread.messages):
            role = msg.__class__.__name__
            content = msg.content if hasattr(msg, "content") else str(msg)
            # Truncate long content
            content_preview = (
                content[:100] + "..." if len(str(content)) > 100 else content
            )
            print(f"  {i+1}. {role}: {content_preview}")


if __name__ == "__main__":
    # Make sure OPENAI_API_KEY is set
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set")
        print("Set it with: export OPENAI_API_KEY=your-key-here")
        exit(1)

    main()
