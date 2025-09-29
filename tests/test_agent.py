import pytest
from pydantic import BaseModel

from ask.core.agent import AgentASK


class InputModel(BaseModel):
    name: str
    age: int


class OutputModel(BaseModel):
    message: str
    count: int


class TestCreateAgentFromFunction:
    @pytest.mark.asyncio
    async def test_create_agent_from_function_str_to_str(self):
        async def upper_func(prompt: str) -> str:
            return prompt.upper()

        agent = AgentASK.create_from_function("upper_agent", upper_func)
        assert isinstance(agent, AgentASK)
        assert agent._input_type is str
        assert agent._output_type is str

        result = await agent.run("hello")
        assert result == "HELLO"

    @pytest.mark.asyncio
    async def test_create_agent_from_function_int_to_int(self):
        async def double_func(prompt: int) -> int:
            return prompt * 2

        agent = AgentASK.create_from_function("double_agent", double_func)
        assert isinstance(agent, AgentASK)
        assert agent._input_type is int
        assert agent._output_type is int

        result = await agent.run(5)
        assert result == 10

    @pytest.mark.asyncio
    async def test_create_agent_from_function_with_stats(self):
        async def identity_func(prompt: str) -> str:
            return prompt

        agent = AgentASK.create_from_function("identity_agent", identity_func)
        assert agent._input_type is str
        assert agent._output_type is str
        result = await agent.run("test")
        assert result == "test"

        # Check stats are updated
        stat = agent.stat
        assert stat._total_requests == 1
        assert stat._duration > 0

    @pytest.mark.asyncio
    async def test_create_agent_from_function_pydantic_models(self):
        async def process_model(prompt: InputModel) -> OutputModel:
            return OutputModel(message=f"Hello {prompt.name}", count=prompt.age * 2)

        agent = AgentASK.create_from_function("process_agent", process_model)
        assert isinstance(agent, AgentASK)
        assert agent._input_type is InputModel
        assert agent._output_type is OutputModel

        input_data = InputModel(name="Alice", age=25)
        result = await agent.run(input_data)
        assert isinstance(result, OutputModel)
        assert result.message == "Hello Alice"
        assert result.count == 50
