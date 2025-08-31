"""
agent.py
"""
from collections.abc import Callable
from typing import Awaitable, List
from attr import dataclass
from pydantic_ai import Agent
from pydantic_ai.usage import UsageLimits, Usage
from pydantic_ai.settings import ModelSettings
from pydantic_ai.messages import ModelMessage

from core.config import Config, load_config
from core.mcp_client import create_mcp_servers
from core.model import create_model
from core.agent_history import make_llm_repack_processor

@dataclass
class AgentStats:
    _usage = Usage()
    
    def _update_stats(self, usage: Usage):
        self._usage = usage 
        
    def __str__(self):
        return f"Total: {self._usage.total_tokens}, Requests: {self._usage.requests}, Details: {self._usage.details}"

class AgentASK:
    _agent: Agent
    _history: list
    _use_mcp_servers: bool
    _stat = AgentStats()
    _repack: Callable[[List[ModelMessage]], Awaitable[List[ModelMessage]]]

    def __init__(self, agent: Agent, use_mcp_servers: bool, repack):
        self._agent = agent
        self._use_mcp_servers = use_mcp_servers
        self._history = []
        self._repack = repack

    async def run_iter(self, iter):
        """Run the agent with the given prompt."""
        if self._use_mcp_servers:
            async with self._agent.run_mcp_servers():
                return await iter()
        return await iter()

    def iter(self, prompt: str):
        async def _iter():
            history = await self._repack(self._history)
            ret = await self._agent.run(prompt, usage_limits=UsageLimits(request_limit=100), message_history=history)
            self._history = ret.all_messages()
            self._stat._update_stats(ret.usage())
            return ret.output
        return _iter

    # wrapper for single shot run
    async def run(self, prompt: str):
        """Run the agent with the given prompt."""
        return await self.run_iter(self.iter(prompt))

    @property
    def stat(self) -> AgentStats:
        """Get the agent's statistics."""
        return self._stat

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
            repack=make_llm_repack_processor(create_model(llm), max_history=config.llm.max_history),
        )

    @classmethod
    def create_from_file(cls, paths: list[str], name: str = "ASK_Agent") -> 'AgentASK':
        """Create a PydanticAI Agent from a config file paths."""
        config = load_config(paths)
        return cls.create_from_config(config, name)
