#!/usr/bin/env python3
"""Blog Post multi-agent pipeline orchestrator."""
import asyncio
import os
import sys

from agent import AgentASK

def _here(*parts: str) -> str:
    """Resolve a path relative to this file's directory."""
    return os.path.join(os.path.dirname(__file__), *parts)

async def run(config_paths: list[str], prompt: str) -> str:
    """Helper to create and run an agent for a single step in the pipeline."""
    agent = AgentASK.create_from_file([_here(p) for p in config_paths])
    return await agent.run(prompt)

async def main(query):
    llm_config = "llm.yaml" 
    research = await run(["research.yaml", llm_config], query)
    outline_prompt = (
        f"topic: {query}\n"
        f"research: {research}\n"
    )
    outline = await run(["outline.yaml", llm_config], outline_prompt)
    post_prompt = (
        f"topic: {query}\n" 
        f"research: {research}\n"
        f"outline: {outline}\n"
    )
    post = await run(["post.yaml", llm_config], post_prompt)
    print(post)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise ValueError("Please provide a query as an argument.")
    asyncio.run(main(" ".join(sys.argv[1:])))
