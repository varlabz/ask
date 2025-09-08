"""Basic REST API for chat using FastAPI and Pydantic AI.

This implements a minimal backend-only version of the Pydantic AI chat example:
- POST /chat: streams NDJSON messages (user then model chunks) and persists them

No UI endpoints are provided.
"""

from __future__ import annotations as _annotations

from contextlib import asynccontextmanager
from datetime import datetime, timezone
import os
import signal
from typing import Annotated
import fastapi
from fastapi import Depends, Form
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from core.agent import AgentASK

# Pydantic models for chat messages
class ChatMessage(BaseModel):
	"""Base model for chat messages."""
	role: str
	timestamp: str
	content: str

class UserMessage(ChatMessage):
	"""User message model."""
	role: str = "user"

class AssistantMessage(ChatMessage):
	"""Assistant message model."""
	role: str = "assistant"

# Define a router to register endpoints without requiring a global app at import time
router = fastapi.APIRouter()

def get_agent(request: fastapi.Request) -> AgentASK: return request.app.state.agent

@router.post("/chat/")
async def post_chat(
	prompt: Annotated[str, Form(...)],
	agent: AgentASK = Depends(get_agent),
) -> StreamingResponse:
	"""Stream chat response and persist conversation.

	Returns an NDJSON stream of ChatMessage objects.
	"""
	async def stream_messages():
		# Create user message using Pydantic model
		user_msg = UserMessage(
			timestamp=datetime.now(tz=timezone.utc).isoformat(),
			content=prompt,
		)
		# Yield the user message as JSON
		yield user_msg.model_dump_json().encode("utf-8") + b"\n"

		# Run the agent
		result = await agent.iter(prompt)()
		
		# Create assistant message using Pydantic model
		assistant_msg = AssistantMessage(
			timestamp=datetime.now(tz=timezone.utc).isoformat(),
			content=result,
		)
		# Yield the assistant message as JSON
		yield assistant_msg.model_dump_json().encode("utf-8") + b"\n"
	
	return StreamingResponse(stream_messages(), media_type="text/plain")

@router.post("/shutdown/")
async def shutdown_server():
    print("Shutdown endpoint called")
    # Send SIGTERM to current process
    os.kill(os.getpid(), signal.SIGTERM)
    return {"message": "Server shutting down..."}

def make_lifespan(agent: AgentASK):
	"""Create a FastAPI lifespan context that uses an externally provided agent."""
	@asynccontextmanager
	async def _lifespan(app: fastapi.FastAPI):
		app.state.agent = agent
		if agent._use_mcp_servers:
			async with agent._agent.run_mcp_servers():
				yield
		yield

	return _lifespan

if __name__ == "__main__":
	import argparse
	import uvicorn
	from core.config import load_config

	parser = argparse.ArgumentParser(description="Run ASK FastAPI server.")
	parser.add_argument(
		"-c",
		"--config",
		type=str,
		action="append",
		help="Path to config yaml (can be used multiple times)",
	)
	args = parser.parse_args()

	config = load_config(args.config or [".ask.yaml"])
	agent = AgentASK.create_from_config(config)
	app = fastapi.FastAPI(lifespan=make_lifespan(agent))
	app.include_router(router)
	uvicorn.run(app, host="127.0.0.1", port=8000)
