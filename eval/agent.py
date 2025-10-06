from textwrap import dedent

from ask.core.config import (
    AgentConfig,
    Config,
    LLMConfig,
    MCPServerConfig,
)

agent_config = AgentConfig(
    instructions=dedent("""
        You are an advanced AI assistant with access to various tools.
        Provide accurate, helpful, and concise responses.
        Must follow instructions precisely.
        """),
)


agent_mcp = {
    "everything": MCPServerConfig(
        command=["npx", "-y", "@modelcontextprotocol/server-everything"]
    )
}


def local(model: str, base_url: str) -> "LLMConfig":
    return LLMConfig(
        model=model,
        base_url=base_url,
        temperature=0.0,
    )


def create_config(
    llm: LLMConfig, instructions: str = "", mcp: dict = agent_mcp
) -> Config:
    config = Config(
        agent=agent_config,
        llm=llm,
        mcp=mcp,
    )
    config.agent.name = f"eval-agent-{llm.model}"
    config.agent.instructions = config.agent.instructions.format(
        extra_instructions=instructions
    )
    return config


# def create_agent[InputT, OutputT](config: Config, input_type: InputT, output_type: OutputT) -> AgentASK[InputT, OutputT]:
#     # copy config to avoid mutating the original
#     cfg = config.model_copy(deep=True)
#     cfg.agent.input_type = input_type
#     cfg.agent.output_type = output_type
#     return AgentASK[InputT, OutputT].create_from_config(cfg)
