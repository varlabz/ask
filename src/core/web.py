#!/usr/bin/env python3
from __future__ import annotations as _annotations

from contextlib import asynccontextmanager
import sys
from typing import Annotated, Literal, AsyncIterator
from datetime import datetime, timezone
from typing import List, Tuple
import httpx
from pydantic import BaseModel
import fastapi
from fastapi.responses import StreamingResponse
from nicegui import ui, app

from src.core.agent import AgentASK

class ChatMessage(BaseModel):
    """Base model for chat messages."""
    role: Literal['user', 'assistant']
    timestamp: str
    content: str
    stat: str | None = None  # Optional statistics for assistant messages

messages: List[ChatMessage] = []

API_BASE_URL = "http://localhost:8000"  # API server URL (overridden by --port at runtime)

def get_agent(request: fastapi.Request) -> AgentASK: return request.app.state.agent

@app.post("/chat/")
async def post_chat(
    prompt: Annotated[str, fastapi.Form(...)],
	agent: AgentASK = fastapi.Depends(get_agent),
) -> StreamingResponse:
    """
    Stream chat response and persist conversation.
    Returns an NDJSON stream of ChatMessage objects.
    """
    async def stream_messages():
        try:
            user_msg = ChatMessage(
                role='user',
                timestamp=datetime.now(tz=timezone.utc).isoformat(),
                content=prompt,
            )
            yield user_msg.model_dump_json().encode("utf-8") + b"\n"
            result = await agent.iter(prompt)()
            stat = str(agent.stat)
            assistant_msg = ChatMessage(
                role='assistant',
                timestamp=datetime.now(tz=timezone.utc).isoformat(),
                content=str(result),
                stat=stat
            )
            yield assistant_msg.model_dump_json().encode("utf-8") + b"\n"
        except BaseException as e:
            print("Error:", e, file=sys.stderr)

    return StreamingResponse(stream_messages(), media_type="text/plain")

@ui.refreshable
def chat_messages() -> None:
    if messages:
        for msg in messages:
            with ui.chat_message(
                name='You' if msg.role == 'user' else 'ASK', 
                sent=(msg.role == 'user'),
                stamp=msg.stat,
            ): 
                ui.markdown(msg.content)
    else:
        with ui.row().classes('justify-center'):
            ui.label('No messages yet')
    ui.run_javascript('window.scrollTo(0, document.body.scrollHeight)')

async def _send(prompt: str) -> AsyncIterator[ChatMessage]:
    """Stream ChatMessage objects from the API as they arrive.

    Yields:
        ChatMessage: Parsed chat messages from the NDJSON stream.
    """
    if not prompt:
        return

    try:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(connect=5.0, read=300.0, write=10.0, pool=5.0)
        ) as client:
            async with client.stream('POST', f'{API_BASE_URL}/chat/', data={'prompt': prompt}) as response:
                response.raise_for_status()
                try:
                    async for line in response.aiter_lines():
                        if not line:
                            continue
                        try:
                            yield ChatMessage.model_validate_json(line)
                        except Exception:
                            ui.notify(f'Error parsing message: {line}', type='negative')
                            continue
                except httpx.ReadTimeout:
                    ui.notify('Timed out waiting for the next message chunk.', type='warning')
    except Exception as e:
        ui.notify(f'Error communicating with API: {e}', type='negative')

@ui.page('/')
async def main():
    async def send(text) -> None:
        value = text.value
        text.value = ''
        spinner.style('visibility: visible')
        async for msg in _send(value):
            messages.append(msg)
            chat_messages.refresh()
        spinner.style('visibility: hidden')

    ui.add_css(r'a:link, a:visited {color: inherit !important; text-decoration: none; font-weight: 500}')
    with ui.footer().classes('bg-black'), ui.column().classes('w-full max-w-3xl mx-auto my-6'):
        with ui.row().classes('w-full no-wrap items-center'):
            spinner = ui.spinner(type='grid').style('visibility: hidden')
            prompt = ui.input(placeholder='prompt').on('keydown.enter', lambda: send(prompt)) \
                .props('rounded outlined input-class=mx-3').classes('flex-grow')

    await ui.context.client.connected()  # chat_messages(...) uses run_javascript which is only possible after connecting
    with ui.column().classes('w-full max-w-2xl mx-auto items-stretch'):
        chat_messages()

def run_web(agent: AgentASK, port: int) -> None:
    main_app_lifespan = app.router.lifespan_context
    @asynccontextmanager
    async def lifespan_wrapper(app):
        app.state.agent = agent
        if agent._use_mcp_servers:
            async with agent._agent.run_mcp_servers():
                async with main_app_lifespan(app) as state:
                    yield state
        else:
            async with main_app_lifespan(app) as state:
                yield state 
    app.router.lifespan_context = lifespan_wrapper
    # Update API base URL to selected port so the UI talks to the correct server
    global API_BASE_URL
    API_BASE_URL = f"http://localhost:{port}"
    ui.run(host="localhost", port=port, title='ASK Chat', dark=None, favicon='🤖', native=True)

if __name__ in {'__main__', '__mp_main__'}:
    import argparse

    parser = argparse.ArgumentParser(description="Run ASK web UI.")
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        action="append",
        help="Path to config yaml (can be used multiple times)",
    )
    parser.add_argument(
        "-p",
        "--port",
        type=int,
        default=8004,
        help="Port to run the app on (default: 8004)",
    )
    args = parser.parse_args()
    agent = AgentASK.create_from_file(args.config or [".ask.yaml"])
    run_web(agent, args.port)