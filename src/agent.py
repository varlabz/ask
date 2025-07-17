"""
agent.py

A streaming agent example using PydanticAI. Run with:
    python src/agent.py --prompt "Your prompt here"
"""
from pydantic_ai import Agent

from config import Config
from mcp_client import create_mcp_servers
from model import create_model

class AgentASK(Agent):
    """Wrapper for Agent class to extract some missing attributes."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.use_mcp_servers = kwargs.get('mcp_servers', None)


def create_agent(config: Config) -> AgentASK:
    """Create a PydanticAI Agent from a Config instance.
    
    Args:
        config: Parsed Config object.
    
    Returns:
        Agent: Configured PydanticAI Agent instance.
    """
    agent = AgentASK(
        name="ASK",
        model=create_model(config.llm),
        system_prompt=config.agent.instructions,
        mcp_servers=create_mcp_servers(config.mcp),
    )
    return agent

async def run_agent(prompt: str, agent: AgentASK) -> str:
    """Run the agent without streaming and return the output as a string.
    Args:
        prompt: The user prompt to pass to the agent.
        agent: The PydanticAI Agent instance to use.
    Returns:
        str: The output from the agent.
    """
    if agent.use_mcp_servers:
        async with agent.run_mcp_servers(): 
            result = await agent.run(prompt)
    else:
        result = await agent.run(prompt)

    return str(result.output.strip())

