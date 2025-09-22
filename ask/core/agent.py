"""
agent.py
"""
from collections.abc import Callable
import time
from typing import Awaitable, Final, List, Generic, TypeVar, Any
from attr import dataclass
from pydantic_ai import Agent
from pydantic_ai.usage import UsageLimits, Usage
from pydantic_ai.settings import ModelSettings
from pydantic_ai.messages import ModelMessage

from .config import Config, LLMConfig, load_config, load_config_dict
from .mcp_client import create_mcp_servers
from .model import create_model
from .agent_history import make_llm_repack_processor, repack_tools_messages

InputT = TypeVar('InputT', default=str)
OutputT = TypeVar('OutputT', default=str)

@dataclass
class AgentStats:
    _usage = Usage()
    _duration = 0
    _total_requests = 0

    def _update_stats(self, usage: Usage, duration: float, ):
        self._usage = usage
        self._duration = duration
        self._total_requests += usage.requests

    def __str__(self):
        return (
            f"total: {self._usage.total_tokens}, tps: {(self._usage.total_tokens or 0)/self._duration:.2f}, "
            f"requests: {self._total_requests}, details: {self._usage.details}"
        )

class AgentASK(Generic[InputT, OutputT]):
    _agent: Agent
    _history: list
    _use_mcp_servers: bool
    _stat = AgentStats()
    _repack: Callable[[List[ModelMessage]], List[ModelMessage]]

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

    def iter(self, prompt: InputT):
        """Create an async iterator for the agent with the given prompt."""
        async def _iter():
            start_time = time.time()
            # Cast to str for current pydantic_ai Agent API expecting a string prompt
            ret = await self._agent.run(str(prompt), usage_limits=UsageLimits(request_limit=100), message_history=self._history)
            end_time = time.time()
            self._history = self._repack(ret.all_messages())
            self._stat._update_stats(ret.usage(), duration=(end_time - start_time), )
            return ret.output
        
        return _iter

    # wrapper for single shot run
    async def run(self, prompt: InputT) -> OutputT:
        """Run the agent with the given prompt."""
        return await self.run_iter(self.iter(prompt))

    @property
    def stat(self) -> AgentStats:
        """Get the agent's statistics."""
        return self._stat

    @classmethod
    def create_from_config(cls, config: Config) -> 'AgentASK[InputT, OutputT]':
        """Create a PydanticAI Agent from a Config instance."""
        llm: Final[LLMConfig] = config.llm
        model_settings = ModelSettings()
        if llm.temperature is not None:
            model_settings["temperature"] = llm.temperature
        if llm.max_tokens is not None:
            model_settings["max_tokens"] = llm.max_tokens
        if llm.timeout is not None:
            model_settings["timeout"] = llm.timeout
            
        return cls(
            agent=Agent(
                name=config.agent.name,
                model=create_model(llm),
                system_prompt=config.agent.instructions,
                mcp_servers=create_mcp_servers(config.mcp),
                model_settings=model_settings,
                output_type=config.agent.output_type,
                retries=3,
                # history_processors=[make_llm_repack_processor(create_model(llm), max_history=config.llm.max_history, keep_last=2)],
            ),
            use_mcp_servers=config.mcp is not None,
            repack=repack_tools_messages if llm.cleanup_history else lambda x: x,
        )

    @classmethod
    def create_from_file(cls, paths: list[str],) -> 'AgentASK[InputT, OutputT]':
        """Create a PydanticAI Agent from a config file paths."""
        config = load_config(paths)
        return cls.create_from_config(config,)

    @classmethod
    def create_from_dict(cls, config_dict: dict, ) -> 'AgentASK[InputT, OutputT]':
        """Create a PydanticAI Agent from a config dictionary."""
        config = load_config_dict(config_dict)
        return cls.create_from_config(config, )
