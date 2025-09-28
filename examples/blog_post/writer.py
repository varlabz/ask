#!/usr/bin/env -S uvx --from git+https://github.com/varlabz/ask ask-run

import os
import sys

from pydantic import Field

from ask.core.agent import AgentASK
from ask.core.agent_context import ContextASK

sys.path.insert(0, os.path.dirname(__file__))
from llm import llm


class WriterInput(ContextASK):
    topic: str = Field(description="The blog post topic")
    research: str = Field(description="The research report")
    outline: str = Field(description="The blog post outline")


writer_agent = AgentASK[WriterInput, str].create_from_dict(
    {
        "agent": {
            "name": "Writer",
            "instructions": f"""
        You are an advanced writer.
        Produce a blog post using the the research report and the blog post outline.
        The blog post will follow the outline and further enrich it with relevant details from the research report.
        The blog post writing style should come across as musings of an intellectual who is trying to examine the topic from various angles.
        Add as much details as possible.

        Input:
        {WriterInput.to_input()}

        Output Format:
        1. Catchy Title.
        2. Engaging introduction that hooks the reader.
        3. TL;DR section with a summary of the main points and key takeaways.
        4. A well-structured blog post.
        5. List of URLs used in the research.
        """,
            "input_type": WriterInput,
            "output_type": str,
        },
        "mcp": {
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
