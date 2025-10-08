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


agent_mcp = {
    "everything": MCPServerConfig(
        command=["npx", "-y", "@modelcontextprotocol/server-everything"]
    )
}
agent_mcp_echo = {
    "echo": MCPServerConfig(
        command=[
            "npx",
            "-y",
            "https://github.com/varlabz/everything-mcp",
            "stdio",
            "--echo",
        ]
    ),
}
agent_mcp_add = {
    "add": MCPServerConfig(
        command=[
            "npx",
            "-y",
            "https://github.com/varlabz/everything-mcp",
            "stdio",
            "--add",
        ]
    ),
}
agent_mcp_longRunningOperation = {
    "longRunningOperation": MCPServerConfig(
        command=[
            "npx",
            "-y",
            "https://github.com/varlabz/everything-mcp",
            "stdio",
            "--longRunningOperation",
        ]
    ),
}
agent_mcp_printEnv = {
    "printEnv": MCPServerConfig(
        command=[
            "npx",
            "-y",
            "https://github.com/varlabz/everything-mcp",
            "stdio",
            "--printEnv",
        ]
    ),
}
agent_mcp_sampleLLM = {
    "sampleLLM": MCPServerConfig(
        command=[
            "npx",
            "-y",
            "https://github.com/varlabz/everything-mcp",
            "stdio",
            "--sampleLLM",
        ]
    ),
}
agent_mcp_getTinyImage = {
    "getTinyImage": MCPServerConfig(
        command=[
            "npx",
            "-y",
            "https://github.com/varlabz/everything-mcp",
            "stdio",
            "--getTinyImage",
        ]
    ),
}
agent_mcp_annotatedMessage = {
    "annotatedMessage": MCPServerConfig(
        command=[
            "npx",
            "-y",
            "https://github.com/varlabz/everything-mcp",
            "stdio",
            "--annotatedMessage",
        ]
    ),
}
agent_mcp_getResourceReference = {
    "getResourceReference": MCPServerConfig(
        command=[
            "npx",
            "-y",
            "https://github.com/varlabz/everything-mcp",
            "stdio",
            "--getResourceReference",
        ]
    ),
}
agent_mcp_elicitation = {
    "elicitation": MCPServerConfig(
        command=[
            "npx",
            "-y",
            "https://github.com/varlabz/everything-mcp",
            "stdio",
            "--elicitation",
        ]
    ),
}
agent_mcp_getResourceLinks = {
    "getResourceLinks": MCPServerConfig(
        command=[
            "npx",
            "-y",
            "https://github.com/varlabz/everything-mcp",
            "stdio",
            "--getResourceLinks",
        ]
    ),
}
agent_mcp_structuredContent = {
    "structuredContent": MCPServerConfig(
        command=[
            "npx",
            "-y",
            "https://github.com/varlabz/everything-mcp",
            "stdio",
            "--structuredContent",
        ]
    ),
}
agent_mcp_listRoots = {
    "listRoots": MCPServerConfig(
        command=[
            "npx",
            "-y",
            "https://github.com/varlabz/everything-mcp",
            "stdio",
            "--listRoots",
        ]
    )
}
agent_mcp_all = [
    # agent_mcp_echo,
    # agent_mcp_add,
    agent_mcp_longRunningOperation,
    agent_mcp_printEnv,
    agent_mcp_sampleLLM,
    agent_mcp_getTinyImage,
    agent_mcp_annotatedMessage,
    agent_mcp_getResourceReference,
    agent_mcp_elicitation,
    agent_mcp_getResourceLinks,
    agent_mcp_structuredContent,
    agent_mcp_listRoots,
]


def local(model: str, base_url: str) -> "LLMConfig":
    return LLMConfig(
        model=model,
        base_url=base_url,
        temperature=0.0,
    )


def create_config(llm: LLMConfig, instructions: str = "", mcp: dict = {}) -> Config:
    config = Config(
        agent=AGENT_CONFIG,
        llm=llm,
        mcp=mcp,
    )
    config.agent.name = f"eval-agent-{llm.model}"
    config.agent.instructions = AGENT_CONFIG.instructions.format(
        instructions=instructions
    )
    return config


# def create_agent[InputT, OutputT](config: Config, input_type: InputT, output_type: OutputT) -> AgentASK[InputT, OutputT]:
#     # copy config to avoid mutating the original
#     cfg = config.model_copy(deep=True)
#     cfg.agent.input_type = input_type
#     cfg.agent.output_type = output_type
#     return AgentASK[InputT, OutputT].create_from_config(cfg)
