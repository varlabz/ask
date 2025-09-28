import os
import sys

from pydantic import Field

from ask.core.agent import AgentASK
from ask.core.agent_context import ContextASK

sys.path.insert(0, os.path.dirname(__file__))
from common import common_mcp, llm


class AssessInput(ContextASK):
    query: str = Field(description="Original user query")
    queries_so_far: list[str] = Field(description="All queries executed so far")


class AssessOutput(ContextASK):
    continue_research: bool = Field(
        description="Whether to continue researching with new queries"
    )
    new_queries: list[str] = Field(
        description="New search queries to execute if continuing; empty if stopping"
    )
    rationale: str = Field(description="Reasoning for continuing or stopping")


assess_agent = AgentASK[AssessInput, AssessOutput].create_from_dict(
    {
        "agent": {
            "name": "Assess",
            "instructions": f"""
            Assess Need for Further Research
            Based on the original query, the search queries performed so far, and the extracted contexts from webpages (available via the memory tool), determine if further research is needed.
            If further research is needed, propose a new list of up to 3 additional search queries. Otherwise, set continue_research to false.

            Input:
            {AssessInput.to_input()}
        """,
            "output_type": AssessOutput,
        },
        "mcp": common_mcp,
        "llm": llm,
    }
)
