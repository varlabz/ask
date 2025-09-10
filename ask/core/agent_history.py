"""
History repacking utilities for PydanticAI agents.

This module provides an LLM-powered history processor to summarize and
compress long conversations while preserving essential context.

TODO: repacking depends on task type

@see https://github.com/Wh1isper/pydantic-ai-history-processor.git 

"""

import pprint
import sys
from textwrap import dedent
from typing import Any, Awaitable, Callable, List
from unittest import result

from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext, ToolOutput
from pydantic_ai.messages import ModelMessage, SystemPromptPart, ModelMessagesTypeAdapter, ModelRequest, UserPromptPart, ModelResponse, TextPart, ToolCallPart, ToolReturnPart, RetryPromptPart
from pydantic_ai.usage import UsageLimits
from pydantic_ai.models import Model
from pydantic_ai.settings import ModelSettings


def dump_messages(messages: List[ModelMessage], file: str) -> List[ModelMessage]:
    """Dump message details to a file."""
    with open(file, "w") as f:
        for m in messages:
            f.write(f"{m.__class__.__qualname__}")
            pprint.pp(vars(m), stream=f, indent=2, width=20)
    return messages

def make_llm_repack_processor(
    model: Model,
    keep_last: int = 3, # use odd number. because last item in list is request and trim part should end with response
    max_history: int = 500,
    max_context_size: int = 100_000,
    system_prompt: str = 
        "You compress conversations into concise, actionable summaries."
        "Remove greetings and small talk. Preserve key facts, decisions, constraints, open questions, and TODOs."
        "Summarize the prior conversation succinctly. Remove chit-chat and repetition."
    ,
) -> Callable[[List[ModelMessage]], Awaitable[List[ModelMessage]]]:
    """
    Create an async history processor that summarizes older messages with an LLM.

    The processor:
    - Preserves the very first request (often includes the system prompt)
    - Keeps the last `keep_last` messages verbatim
    - Summarizes the middle section via the provided `summarizer` agent
    - Injects a single synthetic request containing the summary so the main model
      gets compact context, without leaking the summarizer's own prompts

    Args:
        summarizer: Agent used to generate the summary (ideally a cheaper/smaller model).
        keep_last: Number of the most recent messages to keep verbatim.
        min_messages_before_summarize: Only summarize when history exceeds this count.
        summary_token_limit: Hard cap on summary response tokens to avoid bloat.

    Returns:
        An async callable compatible with pydantic-ai ``history_processors``.
    """
    class RepackResult(BaseModel):
        summary: str = Field(
            ...,
            description="""A summary of the conversation so far""",
        )
        context: str = Field(
            ...,
            description=dedent("""
                The context to continue the conversation with. If applicable based on the current task, this should include:
                - Primary Request and Intent: Capture all of the user's explicit requests and intents in detail
                - Problem Solving: Document problems solved and any ongoing troubleshooting efforts.
                """)
            ,
        )

    summarizer = Agent(
        model=model,
        system_prompt=system_prompt,
        model_settings=ModelSettings(
            temperature=0.0,
        ),
        output_type=ToolOutput(
            type_=RepackResult,
            name="condense",
            description=dedent("""
                Your task is to create a detailed summary of the conversation so far, paying close attention to the user's explicit requests
                and your previous actions. This summary should be thorough in capturing details, patterns,
                and decisions that would be essential for continuing with the conversation and supporting any continuing tasks.
                The user will be presented with a preview of your generated summary and can choose to use it to compact their context window or keep chatting in the current conversation.
                You should consider these to be equivalent to 'condense' when used in a similar context.
                """),
            max_retries=2,
        ),
        retries=2,
    )

    def get_total_tokens(message_history: list[ModelMessage]) -> int:
        return sum(msg.usage.total_tokens for msg in message_history if isinstance(msg, ModelResponse) and msg.usage.total_tokens)

    async def repack(messages: List[ModelMessage]) -> List[ModelMessage]:
        # print(f"### {len(messages)}/{get_total_tokens(messages)}", file=sys.stderr)
        # dump_messages(messages, "tmp/message_dump.txt")
        if max_history <= 0:
            return messages
        
        if len(messages) <= (keep_last + 1):  # +1 for the head
            return messages
        
        if get_total_tokens(messages) < max_context_size:
            return messages

        # Preserve head (usually includes the system prompt)
        head = messages[:1]
         # Preserve the last few messages verbatim
        tail = messages[-keep_last:] if keep_last > 0 else []
        # Middle part to summarize
        middle = messages[1 : len(messages) - len(tail)]
        # Ask summarizer to produce a concise, actionable brief
        # can do it if have tool call and tool call result in the context
        print(f">>> Summarizing {len(middle)} messages", file=sys.stderr)
        summary_result = await summarizer.run(dedent("""
            The user has accepted the condensed conversation summary you generated. Use `condense` to generate a summary and context of the conversation so far.
            This summary covers important details of the historical conversation with the user which has been truncated.
            It's crucial that you respond by ONLY asking the user what you should work on next.
            You should NOT take any initiative or make any assumptions about continuing with work.
            Keep this response CONCISE and wrap your analysis in <summary> and <context> tags to organize your thoughts and ensure you've covered all necessary points.
            Output is max {max_history} words.
            """),
            message_history=middle,
        )
        summary_result = dedent(f"""
            <condense>
            <summary>
            {summary_result.output.summary}
            </summary>
            <context>
            {summary_result.output.context}
            </context>
            </condense>
            """)
        # Inject a single synthetic request with the summary
        summary_message = ModelResponse(
            parts=[TextPart(content=summary_result)]
        )
        ret = head + [summary_message] + tail
        # dump_messages(ret, "tmp/message_dump_new.txt")
        return ret

    return repack

def repack_tools_messages(messages: List[ModelMessage]) -> List[ModelMessage]:
    """Remove tool calls and tool responses; keep surrounding conversation intact.

    Rules:
    - Drop any message whose parts are exclusively tool traffic:
      * ToolCallPart (model decided to call a tool)
      * ToolReturnPart (tool result returned to the model)
      * RetryPromptPart (tool error/retry prompt)
    - Keep all other messages unchanged.
    """
    tool_parts = (ToolCallPart, ToolReturnPart, RetryPromptPart)
    def is_tool_only_message(msg: ModelMessage) -> bool:
        parts = getattr(msg, "parts", None)
        if not parts:
            return False
        return any(isinstance(p, tool_parts) for p in parts)

    # dump_messages(messages, "tmp/message_dump.txt")
    # return dump_messages([m for m in messages if not is_tool_only_message(m)], "tmp/message_dump_filtered.txt")
    return [m for m in messages if not is_tool_only_message(m)]