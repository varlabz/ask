#!/usr/bin/env -S uvx --from git+https://github.com/varlabz/ask ask-run

import os
import sys

from pydantic import Field

from ask.core.agent import AgentASK
from ask.core.agent_context import ContextASK

sys.path.insert(0, os.path.dirname(__file__))
from llm import llm


class SearchInput(ContextASK):
    query: str = Field(description="Search query to execute across various categories")


class SearchOutput(ContextASK):
    urls: list[str] = Field(description="Unique URLs collected from search results")


search_agent = AgentASK[SearchInput, SearchOutput].create_from_dict(
    {
        "agent": {
            "name": "Search",
            "instructions": f"""
            Search and URL Collection
            Use search tool with categories: web,videos,news,it,science,social media.
            For the query, perform searches and collect only unique URLs.

            Input:
            {SearchInput.to_input()}
        """,
            "input_type": SearchInput,
            "output_type": SearchOutput,
        },
        "mcp": {
            "search": {
                "command": [
                    "uvx",
                    "--from",
                    "git+https://github.com/varlabz/searxng-mcp",
                    "mcp-server",
                ],
                "env": {"SEARX_HOST": "http://macook.local:8080"},
            },
        },
        "llm": llm,
    }
)

if __name__ == "__main__":
    import asyncio

    async def main():
        if len(sys.argv) < 2:
            print("Usage: python query_gen.py <query>")
            sys.exit(1)

        query = " ".join(sys.argv[1:])
        output = await search_agent.run(SearchInput(query=query))
        print(output.model_dump_json(indent=2))

    asyncio.run(main())
