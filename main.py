"""
Entry point for the Agent Runtime SDK.

See examples/basic_agent.py for usage examples.
"""

from agent_runtime import AgentRuntime, InMemoryStore


def main():
    print("Agent Runtime SDK v0.1.0")
    print("=======================")
    print()
    print("This is an SDK for building agentic loops with MCP integration.")
    print()
    print("Quick start:")
    print("  1. See examples/basic_agent.py for usage")
    print("  2. Import: from agent_runtime import AgentRuntime, InMemoryStore, MCPServerConfig")
    print("  3. Create agents, threads, and run the agentic loop")
    print()
    print("Documentation: Check the README.md")


if __name__ == "__main__":
    main()
