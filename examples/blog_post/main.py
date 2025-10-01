#!/usr/bin/env -S uvx --from git+https://github.com/varlabz/ask ask-run

# this example show how to create a blog post on a given topic using multiple agents and multiple models
# it demonstrates flexible agent creation using create_from_dict method but keeps the code simple by not using separate files for agent configuration
# it does type verification of inputs and outputs using pydantic models for better clarity and maintainability
# it shows how can use multiple agents and multiple models in a single workflow
# before running this example, make sure to start a local searxng instance and update the SEARX_HOST environment variable in the research_agent configuration below
# also make sure to have the required API keys in place for the LLMs used below

import asyncio
import os
import sys

from ask.core.cache import CacheASK

sys.path.insert(0, os.path.dirname(__file__))
from outline import OutlineInput, outline_agent
from research import Research, research_agent
from score import ScoreInput, score_agent
from writer import WriterInput, writer_agent


async def main(query: str) -> None:
    cache = CacheASK()
    research_result = await research_agent.cache(cache).run(Research(topic=query))
    outline_result = await outline_agent.cache(cache).run(
        OutlineInput(topic=query, research=research_result.report)
    )
    post_result = await writer_agent.cache(cache).run(
        WriterInput(
            topic=query, research=research_result.report, outline=outline_result
        )
    )
    print(post_result)
    score_result = await score_agent.cache(cache).run(
        ScoreInput(topic=query, article=post_result)
    )
    print(score_result, file=sys.stderr)
    cache.clean()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <blog post topic>")
        sys.exit(1)
    query = " ".join(sys.argv[1:])
    asyncio.run(main(query))
