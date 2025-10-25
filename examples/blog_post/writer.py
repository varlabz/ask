#!/usr/bin/env -S uvx --from git+https://github.com/varlabz/ask ask-run

from textwrap import dedent

from llm import llm
from pydantic import BaseModel, Field

from ask.core.agent import AgentASK


class WriterInput(BaseModel):
    topic: str = Field(description="The blog post topic")
    research: str = Field(description="The research report")
    outline: str = Field(description="The blog post outline")


writer_agent = AgentASK[WriterInput, str].create_from_dict(
    {
        "agent": {
            "name": "Writer",
            "instructions": dedent("""
                You are an advanced writer.
                Produce a blog post using the the research report and the blog post outline.
                The blog post will follow the outline and further enrich it with relevant details from the research report.
                The blog post writing style should come across as musings of an intellectual who is trying to examine the topic from various angles.
                Add as much details as possible.
                Add links of the URLs in the body of the blog post where relevant.

                Output Format:
                1. Catchy Title.
                2. Engaging introduction that hooks the reader.
                3. TL;DR section with a summary of the main points and key takeaways.
                4. A well-structured blog post.
                5. List of URLs used in the research.
            """),
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
