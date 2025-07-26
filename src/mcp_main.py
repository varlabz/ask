"""
mcp_main.py CLI entry point for MCP client
"""
import argparse

from config import load_config
from mcp.server.fastmcp import FastMCP
from pydantic_ai import Agent

from agent import create_agent

server = FastMCP('ASK Server')
agent: Agent = None

@server.tool()
async def ask(request: str) -> str:
    """ASK request handler"""
    return await agent.run(request)

def mcp_main() -> None:
    """Main function for MCP CLI entry point.
    """
    parser = argparse.ArgumentParser(description="Run MCP server.")
    parser.add_argument('-c', '--config', type=str, default=".ask.yaml", help='Path to ask config yaml')
    args = parser.parse_args()

    config = load_config(args.config)
    global agent
    agent = create_agent(config)
    server.run()

if __name__ == "__main__":
    mcp_main()
