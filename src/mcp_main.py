"""
mcp_main.py CLI entry point for MCP server
"""
import argparse

from core.config import load_config
from mcp.server.fastmcp import Context, FastMCP
from mcp.server.session import ServerSession
from core.agent import AgentASK

server = FastMCP("ASK Server", log_level="DEBUG")
_agent: AgentASK

@server.tool()
async def ask(request: str, ctx: Context[ServerSession, None]) -> str:
    """ASK request handler with basic error handling."""
    try:
        return await _agent.run(request)
    except Exception as e:  # noqa: BLE001
        # Log the error to the MCP client if context is available
        try:
            await ctx.error(f"ASK tool failed: {e}")
        except Exception:
            # If logging via context fails for any reason, ignore
            pass
        return f"Error: {e}"

def main() -> None:
    """Main function for MCP CLI entry point."""
    global _agent
    parser = argparse.ArgumentParser(description="Run MCP server.")
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        action="append",
        help="Path to config yaml (can be used multiple times)",
    )
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse", "streamable-http"],
        default="stdio",
        help="MCP transport to use (default: stdio)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Port to use for SSE or streamable-http transport (overrides default)",
    )
    args = parser.parse_args()

    # Use default local config if none provided
    config = load_config(args.config or [".ask.yaml"])
    _agent = AgentASK.create_from_config(config)

    transport = args.transport
    if transport in {"streamable-http", "sse"} and args.port is not None:
        # Override default port for SSE or streamable-http transport
        server.settings.port = args.port
        
    server.run(transport=transport)

if __name__ == "__main__":
    main()
