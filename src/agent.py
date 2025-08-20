"""
agent.py
"""
from pydantic_ai.usage import UsageLimits
from pydantic_ai import Agent
from pydantic_ai.settings import ModelSettings

from config import Config, load_config
from mcp_client import create_mcp_servers
from model import create_model

class AgentASK:
    _agent: Agent
    _use_mcp_servers: bool 

    def __init__(self, agent: Agent, use_mcp_servers: bool):
        self._agent = agent
        self._use_mcp_servers = use_mcp_servers

    async def run(self, prompt: str):
        """Run the agent with the given prompt."""
        if self._use_mcp_servers:
            async with self._agent.run_mcp_servers():
                return (await self._agent.run(prompt, usage_limits=UsageLimits(request_limit=300))).output
        return (await self._agent.run(prompt, usage_limits=UsageLimits(request_limit=100))).output

    @classmethod
    def create_from_config(cls, config: Config, name: str = "ASK_Agent") -> 'AgentASK':
        """Create a PydanticAI Agent from a Config instance."""
        llm = config.llm
        model_settings = ModelSettings(
            temperature=llm.temperature,
            max_tokens=llm.max_tokens,
            timeout=llm.timeout,
        )
        return cls(
            agent=Agent(
                name=name,
                model=create_model(llm),
                system_prompt=config.agent.instructions,
                mcp_servers=create_mcp_servers(config.mcp),
                model_settings=model_settings,
                output_type=config.agent.output_type,
            ),
            use_mcp_servers=config.mcp is not None,
        )

    @classmethod
    def create_from_file(cls, paths: list[str], name: str = "ASK_Agent") -> 'AgentASK':
        """Create a PydanticAI Agent from a config file paths."""
        config = load_config(paths)
        return cls.create_from_config(config, name)
