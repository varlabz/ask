import os
import sys

from pydantic import Field

from ask.core.agent import AgentASK
from ask.core.agent_context import ContextASK

sys.path.insert(0, os.path.dirname(__file__))
from common import common_mcp, llm


class ReportInput(ContextASK):
    query: str = Field(description="Original user query")
    urls: list[str] = Field(
        description="Unique URLs used in the research; contexts should be read from the memory tool"
    )


class ReportOutput(ContextASK):
    report: str = Field(
        description="Comprehensive, well-structured report addressing the query"
    )
    summary: str = Field(
        description="Final short summary of key insights and conclusions"
    )


report_agent = AgentASK[ReportInput, ReportOutput].create_from_dict(
    {
        "agent": {
            "name": "Research-5-Report",
            "instructions": f"""
            Step 5 — Write the Comprehensive Report
            Using the gathered contexts (read from the memory tool) and the original query, write a comprehensive, well-structured, and detailed report that addresses the user's query thoroughly. Include all relevant insights and conclusions without extraneous commentary. Provide also a concise short summary of key insights and conclusions.

            Input:
            {ReportInput.to_input()}
        """,
            "output_type": ReportOutput,
        },
        "mcp": common_mcp,
        "llm": llm,
    }
)
