"""
mcp_main.py CLI entry point for MCP client
"""
import argparse

from core.config import load_config
from mcp.server.fastmcp import FastMCP
from core.agent import AgentASK

server = FastMCP('ASK Server')
_agent: AgentASK

@server.tool()
async def ask(request: str) -> str:
    """ASK request handler"""
    return await _agent.run(request)

def main() -> None:
    """Main function for MCP CLI entry point."""
    global _agent
    
    parser = argparse.ArgumentParser(description="Run MCP server.")
    parser.add_argument('-c', '--config', type=str, action='append', help='Path to config yaml (can be used multiple times)')
    args = parser.parse_args()

    # Use default config if none provided
    config = load_config(args.config or [".ask.yaml"])
    
    _agent = AgentASK.create_from_config(config)
    server.run()

if __name__ == "__main__":
    main()
