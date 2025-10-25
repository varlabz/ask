#!/usr/bin/env -S uvx --from git+https://github.com/varlabz/ask ask-run


from textwrap import dedent

from llm import llm_score
from pydantic import BaseModel, Field

from ask.core.agent import AgentASK
from ask.core.context import example


class ScoreInput(BaseModel):
    topic: str = Field(description="The blog post topic")
    article: str = Field(description="The blog post article")


class ScoreOutput(BaseModel):
    class ScoreItem(BaseModel):
        comments: str = Field(description="Detailed qualitative assessment")
        score: int = Field(description="Quantitative score out of 20")

    clarity_readability: ScoreItem = Field(
        description="Evaluate the clarity of the language. Is the writing concise and easily understood by the target audience? Is the sentence structure effective and varied? Is vocabulary precise and appropriate, or is there confusing jargon?"
    )
    depth_accuracy: ScoreItem = Field(
        description="How comprehensively does the article cover the Stated Topic? Does it stay focused? Is there sufficient detail, evidence, and context? Are arguments well-supported by credible facts, data, or expert opinions? (Use external search tools if necessary to verify accuracy). Are there any significant gaps in the information presented relative to the stated topic?"
    )
    structure_organization: ScoreItem = Field(
        description="Analyze the article's logical flow. Does it have a clear introduction that engages the reader, body paragraphs that transition smoothly, and a conclusion that provides a sense of closure? How effective is the use of headings and subheadings in organizing the content?"
    )
    objectivity_bias: ScoreItem = Field(
        description="Scrutinize the article for objectivity. Does it present a balanced view with multiple perspectives? Is the author's tone neutral, or does it betray a particular bias? Are claims presented as fact without sufficient evidence?"
    )
    style_engagement: ScoreItem = Field(
        description="Evaluate the overall writing style. Is it engaging and does it hold the reader's interest? Is the tone appropriate for the subject matter? Assess the effectiveness of the author's voice."
    )
    final_summary: str = Field(
        description="Provide a concise, holistic summary of the article's primary strengths and areas for improvement."
    )
    total_score: int = Field(
        description="Total score out of 100 as sum of the five scores"
    )


score_agent = AgentASK[ScoreInput, ScoreOutput].create_from_dict(
    {
        "agent": {
            "name": "Score",
            "instructions": dedent(f"""
                Role: You are an expert Article Critic.
                Your persona is that of a seasoned editor with a meticulous eye for detail and a profound understanding of journalistic and literary excellence.

                Task: Conduct a comprehensive review of the article provided below,
                evaluating it against its stated topic.
                Your critique must be constructive and professional in tone.

                Format:
                Your analysis must be structured into the five categories listed below.
                For each category, provide:
                A detailed qualitative assessment.
                A quantitative score out of 20.
                After assessing all categories, provide a final summary and calculate the total score out of 100.


                Example of output format for each category:
                {
                example(
                    ScoreOutput(
                        clarity_readability=ScoreOutput.ScoreItem(
                            comments="...", score=15
                        ),
                        depth_accuracy=ScoreOutput.ScoreItem(comments="...", score=18),
                        structure_organization=ScoreOutput.ScoreItem(
                            comments="...", score=17
                        ),
                        objectivity_bias=ScoreOutput.ScoreItem(
                            comments="...", score=16
                        ),
                        style_engagement=ScoreOutput.ScoreItem(
                            comments="...", score=19
                        ),
                        final_summary="...",
                        total_score=85,
                    )
                )
            }
            """),
            "input_type": ScoreInput,
            "output_type": ScoreOutput,
        },
        "llm": llm_score,
    }
)
