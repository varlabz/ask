#!/usr/bin/env -S uvx --from git+https://github.com/varlabz/ask ask-run

import sys
from textwrap import dedent

from llm import llm
from pydantic import BaseModel, Field

from ask.core.agent import AgentASK


class Research(BaseModel):
    topic: str = Field(description="The topic of the blog post")


class ResearchResult(BaseModel):
    report: str = Field(description="The research report")
    urls: list[str] = Field(description="List of URLs used in the research")


research_agent = AgentASK[Research, ResearchResult].create_from_dict(
    {
        "agent": {
            "name": "Research",
            "instructions": dedent("""
                You are an expert research assistant.
                Research the topic for a blog post. A well-done research should include:
                - The basic overview of the topic
                - Historical perspective, if applicable
                - Current opinions on the topic, if applicable
                - Any controversies that might be surrounding the topic
                - Any future developments around the topic
                - Collect list of URLs used in the research

                Use the tools:
                - Search tool to find relevant URLs in categories web,videos,news,social media with no more than 30 results.
                - Converter tool to retrieve content from the URLs and other resources.
            """),
            "input_type": Research,
            "output_type": ResearchResult,
        },
        "mcp": {
            # "search": {
            #     "command": [
            #         "uvx",
            #         "--from",
            #         "git+https://github.com/varlabz/duckduckgo-mcp",
            #         "duckduckgo-mcp",
            #     ]
            # },
            "search": {
                "command": [
                    "uvx",
                    "--from",
                    "git+https://github.com/varlabz/searxng-mcp",
                    "mcp-server",
                ],
                "env": {"SEARX_HOST": "http://macook.local:8080"},
            },
            "converter": {
                "command": [
                    "uvx",
                    "--from",
                    "git+https://github.com/varlabz/markitdown-mhtml.git@mhtml#subdirectory=packages/markitdown-mcp",
                    "markitdown-mcp",
                ],
                "env": {
                    "MARKITDOWN_ENABLE_PLUGINS": "true",
                },
            },
            "sequential_thinking": {
                "command": [
                    "npx",
                    "-y",
                    "@modelcontextprotocol/server-sequential-thinking",
                ],
                "env": {"DISABLE_THOUGHT_LOGGING": "true"},
            },
        },
        "llm": llm,
    }
)

if __name__ == "__main__":
    import asyncio

    from ask.core.cache import CacheASK

    async def main():
        if len(sys.argv) < 2:
            print(f"Usage: {sys.argv[0]} <blog post topic>")
            sys.exit(1)
        query = " ".join(sys.argv[1:])
        cache = CacheASK()
        result = await research_agent.cache(cache).run(Research(topic=query))
        print(result.report)
        print("URLs used in the research:", result.urls, file=sys.stderr)

    asyncio.run(main())
