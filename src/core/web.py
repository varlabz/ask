#!/usr/bin/env python3
from __future__ import annotations as _annotations

import asyncio
from contextlib import asynccontextmanager
from typing import Literal, AsyncIterator
from datetime import datetime, timezone
from typing import List
from pydantic import BaseModel
from nicegui import ui, app

from core.agent import AgentASK

class ChatMessage(BaseModel):
    """Base model for chat messages."""
    role: Literal['user', 'assistant']
    timestamp: str
    content: str
    stat: str | None = None  # Optional statistics for assistant messages

messages: List[ChatMessage] = []
initial_prompt: str | None = None
agent: AgentASK  # Will be set in lifespan

async def _send(prompt: str) -> AsyncIterator[ChatMessage]:
    user_msg = ChatMessage(
        role='user',
        timestamp=datetime.now(tz=timezone.utc).isoformat(),
        content=prompt,
    )
    yield user_msg
    result = await agent.iter(prompt)()
    stat = str(agent.stat)
    assistant_msg = ChatMessage(
        role='assistant',
        timestamp=datetime.now(tz=timezone.utc).isoformat(),
        content=str(result),
        stat=stat
    )
    yield assistant_msg

@ui.refreshable
def chat_messages() -> None:
    if messages:
        for msg in messages:
            with ui.chat_message(
                name='You' if msg.role == 'user' else 'ASK', 
                sent=(msg.role == 'user'),
                stamp=msg.stat,
            ):  ui.markdown(msg.content, 
                    extras=['fenced-code-blocks', 'tables', 'code-friendly', 'strike', 'task_list', 'mermaid']
                )
    else:
        with ui.row().classes('justify-center'):
            ui.label('No messages yet')
    ui.run_javascript('window.scrollTo(0, document.body.scrollHeight)')

@ui.page('/')
async def main():
    async def send(text) -> None:
        if isinstance(text, str):
            value = text
        else:
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

    global initial_prompt        
    if initial_prompt:
        text = initial_prompt
        initial_prompt = None
        await send(text)

def run_web(_agent: AgentASK, port: int, prompt: str | None, reload: bool = True) -> None:
    global agent
    agent = _agent
    
    main_app_lifespan = app.router.lifespan_context
    @asynccontextmanager
    async def lifespan_wrapper(app):
        if agent._use_mcp_servers:
            async with agent._agent.run_mcp_servers():
                async with main_app_lifespan(app) as state:
                    yield state
        else:
            async with main_app_lifespan(app) as state:
                yield state 
    app.router.lifespan_context = lifespan_wrapper

    app.native.window_args['text_select'] = True
    app.native.window_args['zoomable'] = True

    global initial_prompt
    initial_prompt = prompt
    
    try:
        ui.run(host="localhost", port=port, title='ASK Chat', dark=None, favicon='🤖', native=True, reload=reload)
    except (KeyboardInterrupt, asyncio.CancelledError, SystemExit):
        print("Shutting down...")

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
    run_web(agent, args.port, None)