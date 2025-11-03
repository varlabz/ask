"""a set of predefined tools for agents"""

import os
from collections.abc import Callable
from typing import Any


def create_searx_tool(**kwargs) -> Callable:
    from searxng.mcp import search

    os.environ["SEARX_HOST"] = kwargs.get("searxng_host", "http://localhost:8888")
    return search


def create_search_tool(**kwargs) -> Callable:
    from duckduckgo.mcp import search_tool

    return search_tool


def create_convert_tool(**kwargs) -> Callable:
    from markitdown import MarkItDown

    async def convert_to_markdown(uri: str) -> str:
        """Convert a resource described by an http:, https:, file: or data: URI to markdown"""
        return MarkItDown(enable_plugins=True).convert_uri(uri).markdown

    return convert_to_markdown


def get_tool_by_name(name: str, args: dict[str, Any] | None = None) -> Callable:
    """Get a predefined tool by name."""
    tools = {
        "searx": create_searx_tool,
        "convert": create_convert_tool,
        "search": create_search_tool,
    }
    if name in tools:
        return tools[name](**(args or {}))

    raise ValueError(f"Tool '{name}' not found.")
