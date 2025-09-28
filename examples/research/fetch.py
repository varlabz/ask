#!/usr/bin/env -S uvx --from git+https://github.com/varlabz/ask ask-run

import os
import sys
from textwrap import dedent

from pydantic import Field

from ask.core.agent import AgentASK
from ask.core.agent_context import ContextASK

sys.path.insert(0, os.path.dirname(__file__))
from llm import llm


class FetchInput(ContextASK):
    url: str = Field(description="URL to fetch and process")


class FetchOutput(ContextASK):
    filepath: str = Field(description="Path to the fetched file")
    # content: str = Field(description="Content from fetched page")


fetch_agent = AgentASK[FetchInput, FetchOutput].create_from_dict(
    {
        "agent": {
            "name": "Fetch",
            "instructions": dedent(f"""
            Fetch and Convert Content
            For each URL, use the appropriate tool:
            - Use youtube tool for YouTube URLs.
            - Use converter tool to fetch HTML content.

            Input:
            {FetchInput.to_input()}

            Output:
            - create a directory cache if it does not exist
            - save the fetched content to a file in the cache directory
        """),
            "input_type": FetchInput,
            "output_type": FetchOutput,
        },
        "mcp": {
            "youtube": {
                "command": [
                    "npx",
                    "-y",
                    "https://github.com/varlabz/youtube-mcp",
                    "--mcp",
                ]
            },
            "converter": {
                "command": [
                    "uvx",
                    "--from",
                    "git+https://github.com/varlabz/markitdown-mhtml.git@mhtml#subdirectory=packages/markitdown-mcp",
                    "markitdown-mcp",
                ],
                "env": {"MARKITDOWN_ENABLE_PLUGINS": "true"},
            },
            "filesystem": {
                "command": [
                    "npx",
                    "-y",
                    "@modelcontextprotocol/server-filesystem",
                    ".",
                ],
            },
        },
        "llm": llm,
    }
)

if __name__ == "__main__":
    import asyncio

    async def main():
        if len(sys.argv) < 2:
            print("Usage: python fetch.py <url>")
            sys.exit(1)

        with open("log-fetch.jsonl", "w", encoding="utf-8") as log_stream:
            from ask.core.agent_instrumentation import setup_instrumentation

            setup_instrumentation(stream=log_stream)
            url = sys.argv[1]
            output = await fetch_agent.run(FetchInput(url=url))
            print(output.model_dump_json(indent=2))

    asyncio.run(main())
