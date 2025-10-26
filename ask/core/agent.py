"""
agent.py
"""

import asyncio
import inspect
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from textwrap import dedent
from typing import Final, cast, get_args, get_origin

from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.usage import RunUsage, UsageLimits

from ask.core.cache import CacheASK
from ask.core.context import example, schema

from .config import Config, LLMConfig, load_config, load_config_dict
from .mcp_client import create_mcp_servers
from .memory import Memory, NoMemory, memory_factory
from .model import create_model


@dataclass
class AgentStats:
    _usage = RunUsage()
    _duration = 0
    _total_requests = 0

    def _update_stats(
        self,
        usage: RunUsage,
        duration: float,
    ):
        self._usage = usage
        self._duration = duration
        self._total_requests += usage.requests

    def __str__(self):
        return (
            f"total: {self._usage.total_tokens}, tps: {(self._usage.total_tokens or 0) / self._duration:.2f}, "
            f"requests: {self._total_requests}, details: {self._usage.details}"
        )


class AgentASK[InputT: BaseModel | str, OutputT: BaseModel | str]:
    _agent: Agent[InputT, OutputT]
    _name: str
    _memory: Memory
    _use_mcp_servers: bool
    _stat: AgentStats = AgentStats()
    _cache: CacheASK | None = None
    _input_type: type[InputT]
    _output_type: type[OutputT]

    def __init__(
        self,
        agent: Agent,  # type: ignore. if use llm without tools, InputT and OutputT will be generated with another way
        use_mcp_servers: bool,
        memory: Memory,
        input_type: type[InputT],
        output_type: type[OutputT],
    ):
        self._agent = agent  # type: ignore if use llm without tools, InputT and OutputT will be generated with another way
        self._use_mcp_servers = use_mcp_servers
        self._memory = memory
        self._input_type = input_type
        self._output_type = output_type
        if agent.name:
            self._name = agent.name
        else:
            raise ValueError("Agent must have a name")

    async def run_iter(self, iter) -> OutputT:
        """Run the agent with the given prompt."""
        if self._use_mcp_servers:
            async with self._agent:
                return await iter()
        return await iter()

    def _convert_input(self, prompt: InputT) -> str:
        """Convert the input prompt to the expected type."""
        if isinstance(prompt, BaseModel):
            return prompt.model_dump_json()
        return str(prompt)

    async def _agent_run(self, prompt: InputT) -> OutputT:
        """Run the agent with the given prompt."""
        start_time = time.time()
        ret = await self._agent.run(
            self._convert_input(prompt),
            deps=prompt,
            usage_limits=UsageLimits(request_limit=100),
            message_history=self._memory.get(),
        )
        end_time = time.time()
        self._memory.set(ret.all_messages())
        self._stat._update_stats(ret.usage(), duration=(end_time - start_time))
        return ret.output

    def _iter(self, prompt: InputT) -> Callable[[], Awaitable[OutputT]]:
        """Create an async iterator for the agent with the given prompt."""

        async def _cache_run() -> OutputT:
            if self._cache is not None:
                # print(f">>> step: {self._agent.name}", file=sys.stderr)
                async with self._cache.step(self._name, prompt) as (
                    output,
                    set_output,
                ):
                    if output is not None:
                        usage = RunUsage()
                        usage.requests = 1  # Simulate one request
                        self._stat._update_stats(usage, duration=0.001)
                        if isinstance(self._output_type, type) and issubclass(
                            self._output_type, BaseModel
                        ):
                            return cast(
                                OutputT, self._output_type.model_validate(output)
                            )
                        else:
                            return output
                    return set_output(await self._agent_run(prompt))
            else:
                return await self._agent_run(prompt)

        return _cache_run

    # wrapper for single shot run
    async def run(self, prompt: InputT) -> OutputT:
        """Run the agent with the given prompt."""
        return await self.run_iter(self._iter(prompt))

    @property
    def stat(self) -> AgentStats:
        """Get the agent's statistics."""
        return self._stat

    def cache(self, cache: CacheASK) -> "AgentASK[InputT, OutputT]":
        """Cache the agent's execution results."""
        self._cache = cache
        return self

    @staticmethod
    def _prompt_template(
        prompt: str, input_type: type[InputT], output_type: type[OutputT]
    ) -> str:
        return (
            f"{prompt}\n\n"
            "Output schema:\n"
            f"{schema(output_type) if issubclass(output_type, BaseModel) else str(output_type)}\n\n"
            "Must print only in JSON format.\n"
            "No additional text or explanation.\n"
        )

    @classmethod
    def create_from_config(
        cls, config: Config, memory: Memory | None = None
    ) -> "AgentASK[InputT, OutputT]":
        """Create a PydanticAI Agent from a Config instance."""
        llm: Final[LLMConfig] = config.llm
        agent = Agent(
            name=config.agent.name,
            model=create_model(llm),
            system_prompt=config.agent.instructions
            if config.llm.use_tools
            else cls._prompt_template(
                config.agent.instructions,
                config.agent.input_type,
                config.agent.output_type,
            ),
            toolsets=create_mcp_servers(config.mcp),
            output_type=config.agent.output_type if config.llm.use_tools else str,
            retries=3,
            instrument=True,
            deps_type=config.agent.input_type if config.llm.use_tools else str,
        )
        return cls(
            agent=agent,
            use_mcp_servers=config.mcp is not None,
            memory=memory if memory is not None else memory_factory(llm, None),
            input_type=config.agent.input_type,
            output_type=config.agent.output_type,
        )

    @classmethod
    def create_from_file(
        cls,
        paths: list[str],
        memory: Memory | None = None,
    ) -> "AgentASK[InputT, OutputT]":
        """Create a PydanticAI Agent from a config file paths."""
        config = load_config(paths)
        return cls.create_from_config(
            config,
            memory=memory,
        )

    @classmethod
    def create_from_dict(
        cls,
        config_dict: dict,
        memory: Memory | None = None,
    ) -> "AgentASK[InputT, OutputT]":
        """Create a PydanticAI Agent from a config dictionary."""
        config = load_config_dict(config_dict)
        return cls.create_from_config(
            config,
            memory=memory,
        )

    @classmethod
    def create_from_function(
        cls,
        name: str,
        func: Callable[[InputT], Awaitable[OutputT]],
        memory: Memory | None = None,
    ) -> "AgentASK[InputT, OutputT]":
        sig = inspect.signature(func)
        return_annotation = sig.return_annotation
        if get_origin(return_annotation) == Awaitable:
            output_type = get_args(return_annotation)[0]
        else:
            output_type = return_annotation
        params = list(sig.parameters.values())
        input_type = params[0].annotation

        class FunctionAgentASK(AgentASK[InputT, OutputT]):
            def __init__(
                self,
                func: Callable[[InputT], Awaitable[OutputT]],
                input_type: type[InputT],
                output_type: type[OutputT],
            ):
                self._func = func
                self._name = name
                self._stat = AgentStats()
                self._use_mcp_servers = False
                self._input_type = input_type
                self._output_type = output_type
                self._memory = memory if memory is not None else NoMemory()

            async def _agent_run(self, prompt: InputT) -> OutputT:
                start_time = time.time()
                ret = await self._func(prompt)
                end_time = time.time()
                usage = RunUsage(requests=1)
                self._stat._update_stats(
                    usage, duration=max(end_time - start_time, 0.00001)
                )
                return ret

        return FunctionAgentASK(func, input_type=input_type, output_type=output_type)


if __name__ == "__main__":
    from ask.core.config import TraceConfig, load_config
    from ask.core.instrumentation import setup_instrumentation_config

    setup_instrumentation_config(
        load_config(["~/.config/ask/trace.yaml"], type=TraceConfig, key="trace"),
    )

    llm = LLMConfig(
        model="ollama:qwen3:1.7b-q4_K_M",  # gemma3:4b-it-q4_K_M",  #
        base_url="http://bacook.local:11434/v1/",
        temperature=0.0,
        # use_tools=False,
    )

    class AnalysisInput(BaseModel):
        query: str = Field(..., description="The analysis query or prompt")
        response: str = Field(..., description="The content to be analyzed")

    class AnalysisOutput(BaseModel):
        context: str = Field(
            "Summary of the content",
            description=(
                "One sentence summarizing with: "
                "- Main topic/domain."
                "- Key arguments/points."
                "- Intended audience/purpose."
            ),
        )
        keywords: list[str] = Field(
            ...,
            description=(
                "Several specific, distinct keywords that capture key concepts and terminology."
                "Order from most to least important."
                "Don't include keywords that are the name of the speaker or time."
                "At least three keywords, but don't be too redundant."
            ),
        )
        tags: list[str] = Field(
            ...,
            description=(
                "Several broad categories/themes for classification."
                "Include domain, format, and type tags."
                "At least three tags, but don't be too redundant."
            ),
        )

    def create_analysis_agent(
        llm: LLMConfig,
    ) -> AgentASK[AnalysisInput, AnalysisOutput]:
        return AgentASK[AnalysisInput, AnalysisOutput].create_from_config(
            load_config_dict(
                {
                    "agent": {
                        "name": "Analysis",
                        "instructions": dedent(f"""
                    Generate an analysis of the following content by:
                    1. Identifying the most salient keywords (focus on nouns, verbs, and key concepts)
                    2. Extracting core themes and contextual elements
                    3. Creating relevant categorical tags

                    Output example:
                    {
                            example(
                                AnalysisOutput(
                                    context="A brief summary of the content.",
                                    keywords=[
                                        "keyword1",
                                        "keyword2",
                                        "keyword3",
                                        "keyword4",
                                    ],
                                    tags=["tag1", "tag2", "tag3", "tag4", "tag5"],
                                )
                            )
                        }
                """),
                        "input_type": AnalysisInput,
                        "output_type": AnalysisOutput,
                    },
                    "llm": llm,
                }
            ),
            NoMemory(),
        )

    agent = create_analysis_agent(llm)
    res = asyncio.run(
        agent.run(
            AnalysisInput(
                query="What is the capitol of France?",
                response="Paris is the capital of France.",
            )
        )
    )
    print(res)
