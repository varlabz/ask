#!/usr/bin/env python3
"""Basic REST API for chat using FastAPI and Pydantic AI with integrated NiceGUI web interface.

This implements a backend with REST API and web UI:
- POST /chat: streams NDJSON messages (user then model chunks) and persists them
"""

from __future__ import annotations as _annotations

from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Annotated, Final, Literal
import fastapi
from fastapi import Depends, Form
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from .agent import AgentASK

class ChatMessage(BaseModel):
    """Base model for chat messages."""
    role: Literal['user', 'assistant']
    timestamp: str
    content: str

# Define a router to register endpoints without requiring a global app at import time
router: Final[fastapi.APIRouter] = fastapi.APIRouter()

def get_agent(request: fastapi.Request) -> AgentASK: return request.app.state.agent

@router.post("/chat/")
async def post_chat(
	prompt: Annotated[str, Form(...)],
	agent: AgentASK = Depends(get_agent),
) -> StreamingResponse:
	"""
 	Stream chat response and persist conversation.

	Returns an NDJSON stream of ChatMessage objects.
	"""
	async def stream_messages():
		user_msg = ChatMessage(
			role='user',
			timestamp=datetime.now(tz=timezone.utc).isoformat(),
			content=prompt,
		)
		yield user_msg.model_dump_json().encode("utf-8") + b"\n"
		result = await agent.iter(prompt)()
		assistant_msg = ChatMessage(
			role='assistant',
			timestamp=datetime.now(tz=timezone.utc).isoformat(),
			content=result,
		)
		yield assistant_msg.model_dump_json().encode("utf-8") + b"\n"
	
	return StreamingResponse(stream_messages(), media_type="text/plain")


def make_lifespan(agent: AgentASK):
	"""Create a FastAPI lifespan context that uses an externally provided agent."""
	@asynccontextmanager
	async def _lifespan(app: fastapi.FastAPI):
		app.state.agent = agent
		if agent._use_mcp_servers:
			async with agent._agent.run_mcp_servers():
				yield
		else:
			yield

	return _lifespan

if __name__ == "__main__":
	import argparse
	import uvicorn

	parser = argparse.ArgumentParser(description="Run ASK FastAPI server.")
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
		default=8000,
		help="Port to run the API server on (default: 8000)",
	)
	args = parser.parse_args()

	agent = AgentASK.create_from_file(args.config or [".ask.yaml"])
	app = fastapi.FastAPI(lifespan=make_lifespan(agent))
	app.include_router(router)

	uvicorn.run(app, host="localhost", port=args.port)
