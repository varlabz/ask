#!/usr/bin/env -S uvx --from git+https://github.com/varlabz/ask ask-run

import os
import sys

from pydantic import Field

from ask.core.agent import AgentASK
from ask.core.agent_context import ContextASK

sys.path.insert(0, os.path.dirname(__file__))
from llm import llm


class QueryGenInput(ContextASK):
    query: str = Field(description="The user's research query")


class QueryGenOutput(ContextASK):
    queries: list[str] = Field(
        description="Distinct and precise search queries covering different angles"
    )


query_gen_agent = AgentASK[QueryGenInput, QueryGenOutput].create_from_dict(
    {
        "agent": {
            "name": "QueryGen",
            "instructions": f"""
            Query Generation
            Given the user's query, generate a list of 3-5 distinct, precise search queries
            that would help gather comprehensive information on the topic.
            Consider different angles and aspects of the topic to ensure a thorough exploration.

            Input:
            {QueryGenInput.to_input()}
        """,
            "input_type": QueryGenInput,
            "output_type": QueryGenOutput,
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
        output = await query_gen_agent.run(QueryGenInput(query=query))
        print(output.model_dump_json(indent=2))

    asyncio.run(main())
