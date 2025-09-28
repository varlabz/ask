import os
import sys

from pydantic import Field

from ask.core.agent import AgentASK
from ask.core.agent_context import ContextASK

sys.path.insert(0, os.path.dirname(__file__))
from common import llm


class FinalizeInput(ContextASK):
    queries: list[str] = Field(description="All queries executed across iterations")
    urls: list[str] = Field(description="All unique URLs used in research")
    pages_retrieved: int = Field(description="Total pages retrieved successfully")
    pages_rejected: int = Field(description="Total pages rejected/failed")
    report: str = Field(description="The full research report text")
    summary: str = Field(description="The final short summary")


class ResearchOutput(ContextASK):
    report: str = Field(
        description="Comprehensive, well-structured report that addresses the query"
    )
    queries: list[str] = Field(
        description="List of distinct search queries used across iterations"
    )
    urls: list[str] = Field(description="Unique URLs fetched and used in the research")
    pages_retrieved: int = Field(
        description="Total number of pages successfully fetched and processed"
    )
    pages_rejected: int = Field(
        description="Total number of pages rejected or failed during processing"
    )
    summary: str = Field(
        description="Final short summary of key insights and conclusions"
    )


finalize_agent = AgentASK[FinalizeInput, ResearchOutput].create_from_dict(
    {
        "agent": {
            "name": "Finalize",
            "instructions": f"""
            Finalize and Structure Output
            Create a full structured result summarizing:
            - report (verbatim from input)
            - queries (list of all queries made)
            - urls (list of all unique URLs fetched)
            - pages_retrieved
            - pages_rejected
            - summary (verbatim from input)

            Input:
            {FinalizeInput.to_input()}
        """,
            "output_type": ResearchOutput,
        },
        "llm": llm,
    }
)
