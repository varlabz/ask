"""History repacking utilities for PydanticAI agents.

This module provides an LLM-powered history processor to summarize and
compress long conversations while preserving essential context.

Usage:
    from pydantic_ai import Agent
    from core.agent_history import make_llm_repack_processor

    summarizer = Agent(
        model="openai:gpt-4o-mini",
        system_prompt=(
            "You compress conversations into concise, actionable summaries. "
            "Remove greetings and small talk. Preserve key facts, decisions, "
            "constraints, and open questions."
        ),
    )

    history_processors = [make_llm_repack_processor(summarizer)]
    agent = Agent(model="openai:gpt-4o", history_processors=history_processors)
"""

import pprint
import sys
from typing import Any, Awaitable, Callable, List

from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelMessagesTypeAdapter, ModelRequest, UserPromptPart, ModelResponse, TextPart
from pydantic_ai.usage import UsageLimits
from pydantic_ai.models import Model
from pydantic_ai.settings import ModelSettings

def make_llm_repack_processor(
    model: Model,
    keep_last: int = 5,                         # use odd number
    min_messages_before_summarize: int = 55,    # use odd number
    max_history: int = 500,
) -> Callable[[List[ModelMessage]], Awaitable[List[ModelMessage]]]:
    """Create an async history processor that summarizes older messages with an LLM.

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

    summarizer = Agent(
        model=model,
        system_prompt=(
            "You compress conversations into concise, actionable summaries."
            "Remove greetings and small talk. Preserve key facts, decisions, constraints, open questions, and TODOs."
            "Summarize the prior conversation succinctly. Remove chit-chat and repetition."
            f"Output is max ~{max_history} words."
        ),
        model_settings=ModelSettings(
            temperature=0.0
        ),
    )

    def dump_messages(messages: List[ModelMessage], file: str) -> None:
        """Dump message details to a file."""
        with open(file, "w") as f:
            for m in messages:
                f.write(f"{m.__class__.__qualname__}")
                pprint.pp(vars(m), stream=f, indent=2, width=20)

    async def repack(messages: List[ModelMessage]) -> List[ModelMessage]:
        print(f"### {len(messages)} messages...", file=sys.stderr)
        dump_messages(messages, "tmp/message_dump.txt")
        if max_history <= 0:
            return messages
        
        # Only repack when the history is sufficiently long
        if len(messages) <= min_messages_before_summarize:
            return messages

        # Preserve head (usually includes the system prompt)
        head = messages[:1]
        # Preserve the last few messages verbatim
        tail = messages[-keep_last:] if keep_last > 0 else []
        # Middle part to summarize
        middle = messages[1 : len(messages) - len(tail)]
        # Ask summarizer to produce a concise, actionable brief
        summary_result = await summarizer.run(
            "Summarize",
            message_history=middle,
            # usage_limits=UsageLimits(response_tokens_limit=summary_token_limit),
        )
        summary_text = str(summary_result.output).strip()
        # Inject a single synthetic request with the summary
        summary_message = ModelResponse(
            parts=[TextPart(content=f"Conversation summary:\n{summary_text}")]
        )
        ret = head + [summary_message] + tail
        dump_messages(ret, "tmp/message_dump_new.txt")
        return ret

    return repack
