from typing import Dict, List, Optional, Union
from pydantic_ai.mcp import MCPServerSSE, MCPServerStreamableHTTP, MCPServerStdio

from core.config import MCPServerConfig

def create_mcp_servers(mcp_config: Optional[Dict[str, MCPServerConfig]]) -> List[Union[MCPServerSSE, MCPServerStreamableHTTP, MCPServerStdio]]:
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
                    tool_prefix=cfg.tool_prefix
                )
            )
        elif transport == "http" or transport == "streamable-http":
            if not cfg.url:
                raise ValueError(f"HTTP transport requires 'url' for server '{name}'")
            servers.append(
                MCPServerStreamableHTTP(
                    url=cfg.url, 
                    tool_prefix=cfg.tool_prefix
                    )
                )
        elif transport == "stdio":
            if not command:
                raise ValueError(f"Stdio transport requires 'command' for server '{name}'")
            servers.append(
                MCPServerStdio(
                    command[0],
                    args=command[1:],
                    tool_prefix=cfg.tool_prefix,
                    cwd=cfg.cwd,
                    env=cfg.env
                )
            )
        else:
            raise ValueError(f"Unknown MCP transport '{transport}' for server '{name}'")
        
    return servers
