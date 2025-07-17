"""
agent.py

A streaming agent example using PydanticAI. Run with:
    python src/agent.py --prompt "Your prompt here"
"""
from pydantic_ai import Agent
from pydantic_ai.settings import ModelSettings

from config import Config
from mcp_client import create_mcp_servers
from model import create_model


class AgentASK(Agent):
    """Agent wrapper with MCP server support."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.use_mcp_servers = bool(kwargs.get('mcp_servers'))


def create_agent(config: Config) -> AgentASK:
    """Create a PydanticAI Agent from a Config instance."""
    llm = config.llm
    model_settings = ModelSettings(
        temperature=llm.temperature,
        max_tokens=llm.max_tokens,
        timeout=llm.timeout,
    )
    return AgentASK(
        name="ASK",
        model=create_model(llm),
        system_prompt=config.agent.instructions,
        mcp_servers=create_mcp_servers(config.mcp),
        model_settings=model_settings,
    )

async def run_agent(prompt: str, agent: AgentASK) -> str:
    """Run the agent and return the output."""
    if agent.use_mcp_servers:
        async with agent.run_mcp_servers():
            result = await agent.run(prompt)
            return result.output.strip()

    result = await agent.run(prompt)
    return result.output.strip()

