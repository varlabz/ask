#!/usr/bin/env -S uvx --from git+https://github.com/varlabz/ask ask-run 

import os
import sys

from pydantic import Field

from ask.core.agent import AgentASK
from ask.core.agent_context import ContextASK

sys.path.insert(0, os.path.dirname(__file__))
from llm import llm

class OutlineInput(ContextASK):
    topic: str = Field(description="The blog post topic")
    research: str = Field(description="The research report")

outline_agent = AgentASK[OutlineInput, str].create_from_dict({
"agent": {
    "name": "Outline",
    "instructions": f"""
        You are an expert writer. 
        Examine the initial topic and the research report summary and come up with an outline for a blog post.
        The outline will weave together the following details:
        - The basic overview of the topic
        - Historical perspective, if applicable
        - Current opinions on the topic, if applicable
        - Any controversies that might be surrounding the topic
        - Any future developments around the topic
        The format of the outline is informal, aiming to translate the dry research report summary into an accessible and entertaining read.
        Use sequential thinking to connect the dots and create a compelling narrative.
        
        Input:
        {OutlineInput.to_input()}
        """,
    "input_type": OutlineInput,
    "output_type": str,
},
"mcp": {
"sequential_thinking": {
        "command": ["npx", "-y", "@modelcontextprotocol/server-sequential-thinking"],
        "env": {
            "DISABLE_THOUGHT_LOGGING": "true"
        },
    },
},
"llm": llm,
})
