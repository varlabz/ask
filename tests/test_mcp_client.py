import os
import sys
import pytest

from pydantic_ai.mcp import MCPServerSSE, MCPServerStreamableHTTP, MCPServerStdio

from ask.core.mcp_client import create_mcp_servers
from ask.core.config import MCPServerConfig

class TestCreateMCPServers:
    def test_sse_positive(self):
        config = {
            "sse_server": MCPServerConfig(
                enabled=True,
                transport="sse",
                url="http://localhost:3001/sse",
            )
        }
        clients = create_mcp_servers(config)
        assert len(clients) == 1
        assert isinstance(clients[0], MCPServerSSE)
        assert clients[0].url == "http://localhost:3001/sse"

    def test_http_positive(self):
        config = {
            "http_server": MCPServerConfig(
                enabled=True,
                transport="http",
                url="http://localhost:8000/mcp",
            )
        }
        clients = create_mcp_servers(config)
        assert len(clients) == 1
        assert isinstance(clients[0], MCPServerStreamableHTTP)
        assert clients[0].url == "http://localhost:8000/mcp"

    def test_stdio_positive(self):
        config = {
            "stdio_server": MCPServerConfig(
                enabled=True,
                transport="stdio",
                command=["python", "server.py"],
            )
        }
        clients = create_mcp_servers(config)
        assert len(clients) == 1
        assert isinstance(clients[0], MCPServerStdio)
        assert clients[0].args == ["server.py"]

    def test_disabled_server(self):
        config = {
            "disabled_server": MCPServerConfig(
                enabled=False,
                transport="sse",
                url="http://localhost:3001/sse",
            )
        }
        clients = create_mcp_servers(config)
        assert clients == []

    def test_missing_url_sse(self):
        config = {
            "bad_sse": MCPServerConfig(
                enabled=True,
                transport="sse",
                url=None,
            )
        }
        with pytest.raises(ValueError, match="SSE transport requires 'url'"):
            create_mcp_servers(config)

    def test_missing_url_http(self):
        config = {
            "bad_http": MCPServerConfig(
                enabled=True,
                transport="http",
                url=None,
            )
        }
        with pytest.raises(ValueError, match="HTTP transport requires 'url'"):
            create_mcp_servers(config)

    def test_missing_command_stdio(self):
        config = {
            "bad_stdio": MCPServerConfig(
                enabled=True,
                transport="stdio",
                command=None,
            )
        }
        with pytest.raises(ValueError, match="Stdio transport requires 'command'"):
            create_mcp_servers(config)

