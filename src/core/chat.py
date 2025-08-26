import asyncio
from typing import Final, Optional
from core.agent import AgentASK
from prompt_toolkit import PromptSession
from prompt_toolkit.history import InMemoryHistory

async def chat(agent: AgentASK, initial_prompt: Optional[str] = None):
    """Interactive chat with the agent."""
    session: Final = PromptSession(history=InMemoryHistory())

    async def get_input(prompt: str) -> str:
        return await session.prompt_async(prompt)

    if initial_prompt:
        response = await agent.iter(initial_prompt)()
        print(response)

    while True:
        try:
            prompt = (await get_input(">> ")).strip()
            if not prompt:
                continue
            if prompt.lower() in ["/exit", "/quit"]:
                break

            response = await agent.iter(prompt)()
            print(response)
        except (KeyboardInterrupt, EOFError):
            break
    print("bye")
