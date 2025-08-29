import asyncio
from typing import Final, Optional
from core.agent import AgentASK
from prompt_toolkit import PromptSession
from prompt_toolkit.history import InMemoryHistory

# TODO maybe to use stream from LLM?
async def _stream_print(text: str, sleep: float = 0.001, yield_every: int = 32) -> None:
    """Print text character-by-character, fast, emulating streaming.

    Args:
        text: The full response to print.
        sleep: Optional per-character delay (0 for as-fast-as-possible).
        yield_every: Yield to the event loop every N characters to stay responsive.
    """
    for i, ch in enumerate(text):
        print(ch, end="", flush=True)
        if sleep > 0:
            await asyncio.sleep(sleep)
        elif (i + 1) % yield_every == 0:
            # No artificial delay, just yield to keep the loop responsive
            await asyncio.sleep(0)
    print()  # Ensure newline at the end

async def chat(agent: AgentASK, initial_prompt: Optional[str] = None):
    """Interactive chat with the agent."""
    session: Final = PromptSession(history=InMemoryHistory())
    counter = 0

    async def get_input(prompt: str) -> str:
        return await session.prompt_async(prompt)

    if initial_prompt:
        response = await agent.iter(initial_prompt)()
        await _stream_print(str(response))

    while True:
        try:
            prompt = (await get_input(f"[{counter}]: ")).strip()
            if not prompt:
                continue
            if prompt.lower() in ["/exit", "/quit"]:
                break

            response = await agent.iter(prompt)()
            await _stream_print(str(response))
            counter += 1
        except (KeyboardInterrupt, EOFError):
            break
    print("bye")
