"""
mcp_main.py CLI entry point for MCP server
"""
import argparse
from typing import Final

from mcp.server.fastmcp import Context, FastMCP
from mcp.server.session import ServerSession

from .core.agent import AgentASK
from .core.config import Config, load_config, ServerConfig

"""Main function for MCP CLI entry point."""
parser = argparse.ArgumentParser(description="Run MCP server.")
parser.add_argument(
    "-c",
    "--config",
    type=str,
    action="append",
    help="Path to config yaml (can be used multiple times)",
)
args = parser.parse_args()

config: Final[Config] = load_config(args.config or [".ask.yaml"])
agent: Final[AgentASK] = AgentASK.create_from_config(config)
server_config: Final[ServerConfig] = config.server or ServerConfig()
server: Final[FastMCP] = FastMCP(
        name=server_config.name,
        instructions=server_config.instructions,
        stateless_http=True,
        streamable_http_path="/",
        # json_response=True,
        debug=server_config.debug,
        log_level=server_config.log_level,
        port=server_config.port,
    )

@server.resource("info://server/description")
def server_description() -> str:
    """The server description/instructions."""
    desc = server_config.instructions
    if desc and desc.strip(): 
        return desc
    
    return f"{server_config.name} — no description configured."

@server.tool()
async def ask(request: str, ctx: Context[ServerSession, None]) -> str:
    """ASK request handler"""
    try:
        return await agent.run(request)
    except Exception as e:  # noqa: BLE001
        # Log the error to the MCP client if context is available
        try:
            await ctx.error(f"ASK tool failed: {e}")
        except Exception:
            # If logging via context fails for any reason, ignore
            pass
        return f"Error: {e}"

def main() -> None:
    server.run(transport=server_config.transport)

if __name__ == "__main__":
    main()
