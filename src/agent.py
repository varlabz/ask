"""
agent.py
"""
from functools import singledispatch
from pydantic_ai import Agent
from pydantic_ai.settings import ModelSettings

from config import Config, load_config
from mcp_client import create_mcp_servers
from model import create_model



class AgentASK(Agent):
    """Agent wrapper with MCP server support."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.use_mcp_servers = bool(kwargs.get('mcp_servers'))

    async def run(self, prompt: str) -> str:
        """Run the agent with the given prompt."""
        if self.use_mcp_servers:
            async with self.run_mcp_servers():
                return (await super().run(prompt)).output
        return (await super().run(prompt)).output

    @staticmethod
    @singledispatch
    def create(config: 'Config') -> 'AgentASK':
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
            output_type=config.agent.output_type,
        )

    @staticmethod
    @create.register
    def _(config_path: str) -> 'AgentASK':
        """Create a PydanticAI Agent from a config file path."""
        config = load_config(config_path)
        return AgentASK.create(config)


