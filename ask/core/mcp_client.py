from pydantic_ai.mcp import MCPServerSSE, MCPServerStdio, MCPServerStreamableHTTP

from .config import MCPServerConfig


def create_mcp_servers(
    mcp_config: dict[str, MCPServerConfig] | None,
) -> list[MCPServerSSE | MCPServerStreamableHTTP | MCPServerStdio]:
    """Create MCP client/server objects from config."""
    if not mcp_config:
        return []

    servers = []
    for name, cfg in mcp_config.items():
        if not cfg.enabled:
            continue

        transport = cfg.transport.lower()
        command = cfg.command or []
        if transport == "sse":
            if not cfg.url:
                raise ValueError(f"SSE transport requires 'url' for server '{name}'")
            servers.append(
                MCPServerSSE(
                    url=cfg.url,
                    tool_prefix=cfg.tool_prefix,
                    timeout=60,
                )
            )
        elif transport == "http":
            if not cfg.url:
                raise ValueError(f"HTTP transport requires 'url' for server '{name}'")
            servers.append(
                MCPServerStreamableHTTP(
                    url=cfg.url,
                    tool_prefix=cfg.tool_prefix,
                    timeout=60,
                )
            )
        elif transport == "stdio":
            if not command:
                raise ValueError(
                    f"Stdio transport requires 'command' for server '{name}'"
                )
            servers.append(
                MCPServerStdio(
                    command[0],
                    args=command[1:],
                    tool_prefix=cfg.tool_prefix,
                    cwd=cfg.cwd,
                    env=cfg.env,
                    timeout=60,
                )
            )
        else:
            raise ValueError(f"Unknown MCP transport '{transport}' for server '{name}'")

    return servers
