from textwrap import dedent
from typing import Final

from ask.core.config import (
    AgentConfig,
    Config,
    LLMConfig,
    MCPServerConfig,
)

AGENT_CONFIG: Final[AgentConfig] = AgentConfig(
    instructions=dedent("""
        You are an advanced AI assistant with access to various tools.
        Provide accurate, helpful, and concise responses.
        Follow instructions precisely.
        {instructions}
        """),
)


def make_llm_config(model: str, base_url: str | None) -> "LLMConfig":
    return LLMConfig(
        model=model,
        base_url=base_url,
        temperature=0.0,
    )


def create_config(
    llm: LLMConfig,
    instructions: str = "",
    mcp: dict[str, MCPServerConfig] | None = None,
) -> Config:
    config = Config(
        agent=AGENT_CONFIG.model_copy(deep=True),
        llm=llm,
        mcp=mcp,
    )
    config.agent.name = f"eval-agent-{llm.model}"
    config.agent.instructions = AGENT_CONFIG.instructions.format(
        instructions=instructions
    )
    return config
