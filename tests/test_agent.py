import pytest

from ask.core.agent import AgentASK


class TestCreateAgentFromFunction:
    @pytest.mark.asyncio
    async def test_create_agent_from_function_str_to_str(self):
        async def upper_func(prompt: str) -> str:
            return prompt.upper()

        agent = AgentASK.create_from_function(upper_func)
        assert isinstance(agent, AgentASK)

        result = await agent.run("hello")
        assert result == "HELLO"

    @pytest.mark.asyncio
    async def test_create_agent_from_function_int_to_int(self):
        async def double_func(prompt: int) -> int:
            return prompt * 2

        agent = AgentASK.create_from_function(double_func)
        assert isinstance(agent, AgentASK)

        result = await agent.run(5)
        assert result == 10

    @pytest.mark.asyncio
    async def test_create_agent_from_function_with_stats(self):
        async def identity_func(prompt: str) -> str:
            return prompt

        agent = AgentASK.create_from_function(identity_func)
        result = await agent.run("test")
        assert result == "test"

        # Check stats are updated
        stat = agent.stat
        assert stat._total_requests == 1
        assert stat._duration > 0
