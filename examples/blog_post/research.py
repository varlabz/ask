#!/usr/bin/env -S uvx --from git+https://github.com/varlabz/ask ask-run

import os
import sys

from pydantic import Field

from ask.core.agent import AgentASK
from ask.core.agent_context import ContextASK

sys.path.insert(0, os.path.dirname(__file__))
from llm import llm


class Research(ContextASK):
    topic: str = Field(description="The topic of the blog post")


class ResearchResult(ContextASK):
    report: str = Field(description="The research report")
    urls: list[str] = Field(description="List of URLs used in the research")


research_agent = AgentASK[Research, ResearchResult].create_from_dict(
    {
        "agent": {
            "name": "Research",
            "instructions": f"""
        You are an expert research assistant.
        Research the topic for a blog post. A well-done research should include:
        - The basic overview of the topic
        - Historical perspective, if applicable
        - Current opinions on the topic, if applicable
        - Any controversies that might be surrounding the topic
        - Any future developments around the topic
        - Collect list of URLs used in the research

        Use the tools:
        - Search tool to find relevant URLs in categories general,videos,news,social_media with number of results 30.
        - Fetch tool to retrieve content from the URLs with max length 100000.

        Input:
        {Research.to_input()}
    """,
            "input_type": Research,
            "output_type": ResearchResult,
        },
        "mcp": {
            "search": {
                "command": [
                    "uvx",
                    "--from",
                    "git+https://github.com/varlabz/duckduckgo-mcp",
                    "duckduckgo-mcp",
                ]
            },
            "fetch": {"command": ["uvx", "mcp-server-fetch", "--ignore-robots-txt"]},
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

    from ask.core.agent_cache import CacheASK

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
