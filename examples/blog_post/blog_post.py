#!/usr/bin/env python3
"""Blog Post multi-agent pipeline orchestrator."""
import asyncio
import os
import sys
import logging

from agent import AgentASK

logger = logging.getLogger(__name__)

DEFAULT_LLM = "~/.config/ask/llm.yaml"
LLM = "llm.yaml"

def _here(*parts: str) -> str:
    """Path helper relative to this file."""
    return os.path.join(os.path.dirname(__file__), *parts)

async def run(config_paths: list[str], prompt: str, step: str) -> str:
    """Run a single pipeline step and optionally log prompt/response.

    Args:
        config_paths: Sequence of config file names relative to this dir.
        prompt: Input prompt for the agent.
        step: Step label used for logging.
    Returns:
        The agent response text.
    """
    print(f"### {step}", file=sys.stderr)
    agent = AgentASK.create_from_file([DEFAULT_LLM] + [_here(p) for p in config_paths], name=step)
    response = await agent.run(prompt)
    logger.info(f"Step: {step}\nPrompt:\n{prompt}\nResponse:\n{response}")
    return response

async def main(query: str) -> None:
    """Execute all steps then print final post."""
    research = await run(["research.yaml", LLM], 
        query, 
        step="research")
    outline = await run(["outline.yaml", LLM],
        f"# topic: {query}\n# research: {research}\n",
        step="outline")
    post = await run(["post.yaml", LLM],
        f"# topic: {query}\n# research: {research}\n# outline: {outline}\n",
        step="post")
    print(post)
    # NOTA BENE: use another model for scoring
    score = await run(["critic.yaml"],
        f"# topic: {query}\n# article: {post}\n", 
        step="score")
    print(score)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Blog Post multi-agent pipeline orchestrator.")
    parser.add_argument("query", nargs="+", help="Query for the blog post topic.")
    parser.add_argument("--logs", action="store_true", help="Save responses to log file.")
    args = parser.parse_args()

    if args.logs:
        log_path = os.path.splitext(os.path.basename(__file__))[0] + ".log"
        logging.basicConfig(
            filename=log_path,
            filemode="a",
            format="%(asctime)s %(levelname)s %(message)s",
            level=logging.INFO,
        )
    else:
        logger.addHandler(logging.NullHandler())

    asyncio.run(main(" ".join(args.query)))

