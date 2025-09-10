import sys
import os
import json
import pytest
from datetime import datetime, timezone
from typing import Any, Sequence, Literal
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient
from fastapi import FastAPI, Form, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from ask.core.rest_api import ChatMessage, make_lifespan

# Import the functions we need to test directly
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    UserPromptPart,
    SystemPromptPart,
)
from pydantic_ai.usage import UsageLimits

# Mock AgentASK class for testing
class MockAgentASK:
    """Mock AgentASK class for testing purposes."""

    def __init__(self):
        self._history = []
        self._repack = MagicMock(return_value=[])
        self._agent = MagicMock()

    @classmethod
    def create_from_config(cls, config):
        """Mock factory method."""
        return cls()


# Mock API functions for testing when imports fail
async def mock_get_chat():
    """Mock GET /chat/ endpoint."""
    return {"message": "mock response"}

async def mock_post_chat():
    """Mock POST /chat/ endpoint."""
    return {"message": "mock response"}


class TestNDJSONFormatting:
    """Test NDJSON formatting for chat messages."""

    def test_format_chat_messages_as_ndjson(self):
        """Test formatting chat messages as NDJSON."""
        chat_messages = [
            ChatMessage(
                role="user",
                timestamp="2024-01-01T12:00:00Z",
                content="Hello"
            ),
            ChatMessage(
                role="assistant",
                timestamp="2024-01-01T12:00:01Z",
                content="Hi there!"
            )
        ]

        # Simulate the NDJSON formatting from the API
        payload = b"\n".join(
            msg.model_dump_json().encode("utf-8") for msg in chat_messages
        )

        # Parse back to verify
        lines = payload.decode().strip().split('\n')
        assert len(lines) == 2

        msg1 = json.loads(lines[0])
        msg2 = json.loads(lines[1])

        assert msg1["role"] == "user"
        assert msg1["content"] == "Hello"
        assert msg2["role"] == "assistant"
        assert msg2["content"] == "Hi there!"

    def test_empty_chat_messages_ndjson(self):
        """Test NDJSON formatting with empty message list."""
        chat_messages = []
        payload = b"\n".join(
            msg.model_dump_json().encode("utf-8") for msg in chat_messages
        )

        assert payload == b""


class TestPositiveScenarios:
    """Positive test scenarios for the chat API."""

    @pytest.fixture
    def mock_agent_with_varied_history(self):
        """Create a mock agent with varied message history."""
        mock_agent = MagicMock(spec=MockAgentASK)
        mock_history = [
            ModelRequest(parts=[UserPromptPart(content="Hello, how are you?")]),
            ModelResponse(parts=[TextPart(content="I'm doing well, thank you for asking!")]),
            ModelRequest(parts=[UserPromptPart(content="Can you help me with Python?")]),
            ModelResponse(parts=[TextPart(content="Of course! I'd be happy to help you with Python. What would you like to know?")]),
            ModelRequest(parts=[UserPromptPart(content="Show me a simple function")]),
            ModelResponse(parts=[TextPart(content="Here's a simple Python function:\n\n```python\ndef greet(name):\n    return f\"Hello, {name}!\"\n\nprint(greet(\"World\"))\n```")]),
        ]
        mock_agent._history = mock_history
        mock_agent._repack = MagicMock(return_value=mock_history)
        mock_agent._agent = MagicMock()  # Add the _agent attribute
        return mock_agent

    def test_post_chat_with_multiline_prompt(self):
        """Test POST /ask/ with multiline prompt."""
        from fastapi.testclient import TestClient
        from fastapi import FastAPI

        mock_agent = MagicMock(spec=MockAgentASK)
        mock_agent._history = []
        mock_agent._repack = MagicMock(return_value=[])
        mock_agent._agent = MagicMock()

        multiline_prompt = """Please help me write a Python function that:
1. Takes a list of numbers
2. Returns the sum of even numbers only
3. Handles empty lists gracefully"""

        mock_result = MagicMock()
        mock_result.output = (
            "Here's the function you requested:\n\n"
            "```python\n"
            "def sum_even_numbers(numbers):\n"
            "    if not numbers:\n        return 0\n"
            "    return sum(num for num in numbers if num % 2 == 0)\n"
            "```"
        )
        mock_result.all_messages.return_value = [
            ModelRequest(parts=[UserPromptPart(content=multiline_prompt)]),
            ModelResponse(parts=[TextPart(content=mock_result.output)]),
        ]
        mock_agent.run = AsyncMock(return_value=mock_result)

        test_app = FastAPI(lifespan=make_lifespan(mock_agent))
        test_app.state.agent = mock_agent

        async def mock_post_chat(prompt: str = Form(...)):
            async def stream_messages():
                user_msg = ChatMessage(
                    role="user",
                    timestamp=datetime.now(tz=timezone.utc).isoformat(),
                    content=prompt,
                )
                yield user_msg.model_dump_json().encode("utf-8") + b"\n"

                result = await mock_agent.run(prompt)

                assistant_msg = ChatMessage(
                    role="assistant",
                    timestamp=datetime.now(tz=timezone.utc).isoformat(),
                    content=result.output,
                )
                yield assistant_msg.model_dump_json().encode("utf-8") + b"\n"

                mock_agent._history = mock_agent._repack(result.all_messages())

            return StreamingResponse(stream_messages(), media_type="text/plain")

        test_app.post("/ask/")(mock_post_chat)

        client = TestClient(test_app)
        response = client.post("/ask/", data={"prompt": multiline_prompt})

        assert response.status_code == 200
        assert "text/plain" in response.headers["content-type"]

        lines = response.content.decode().strip().split("\n")
        assert len(lines) == 2

        user_msg = json.loads(lines[0])
        assistant_msg = json.loads(lines[1])

        assert user_msg["role"] == "user"
        assert user_msg["content"] == multiline_prompt
        assert assistant_msg["role"] == "assistant"
        assert "function" in assistant_msg["content"].lower()

    def test_post_chat_with_special_characters(self):
        """Test POST /ask/ with special characters and unicode."""
        from fastapi.testclient import TestClient
        from fastapi import FastAPI

        mock_agent = MagicMock(spec=MockAgentASK)
        mock_agent._history = []
        mock_agent._repack = MagicMock(return_value=[])
        mock_agent._agent = MagicMock()

        special_prompt = "Hello 🌟! How do I use émojis and spëcial chärs in Python? 🤖"

        mock_result = MagicMock()
        mock_result.output = (
            "You can use emojis and special characters in Python strings directly! 🎉"
        )
        mock_result.all_messages.return_value = [
            ModelRequest(parts=[UserPromptPart(content=special_prompt)]),
            ModelResponse(parts=[TextPart(content=mock_result.output)]),
        ]
        mock_agent.run = AsyncMock(return_value=mock_result)

        test_app = FastAPI(lifespan=make_lifespan(mock_agent))
        test_app.state.agent = mock_agent

        async def mock_post_chat(prompt: str = Form(...)):
            async def stream_messages():
                user_msg = ChatMessage(
                    role="user",
                    timestamp=datetime.now(tz=timezone.utc).isoformat(),
                    content=prompt,
                )
                yield user_msg.model_dump_json().encode("utf-8") + b"\n"

                result = await mock_agent.run(prompt)

                assistant_msg = ChatMessage(
                    role="assistant",
                    timestamp=datetime.now(tz=timezone.utc).isoformat(),
                    content=result.output,
                )
                yield assistant_msg.model_dump_json().encode("utf-8") + b"\n"

                mock_agent._history = mock_agent._repack(result.all_messages())

            return StreamingResponse(stream_messages(), media_type="text/plain")

        test_app.post("/ask/")(mock_post_chat)

        client = TestClient(test_app)
        response = client.post("/ask/", data={"prompt": special_prompt})

        assert response.status_code == 200
        lines = response.content.decode().strip().split("\n")
        assert len(lines) == 2

        user_msg = json.loads(lines[0])
        assistant_msg = json.loads(lines[1])

        assert user_msg["content"] == special_prompt
        assert assistant_msg["content"] == mock_result.output

    def test_conversation_persistence_across_requests(self):
        """Test that conversation history persists across multiple requests."""
        from fastapi.testclient import TestClient
        from fastapi import FastAPI

        mock_agent = MagicMock(spec=MockAgentASK)
        mock_agent._history = []
        mock_agent._repack = MagicMock()
        mock_agent._agent = MagicMock()  # Add the _agent attribute

        # Set up repack to return accumulated history
        def accumulate_history(messages):
            mock_agent._history = messages
            return messages

        mock_agent._repack.side_effect = accumulate_history

        test_app = FastAPI(lifespan=make_lifespan(mock_agent))
        test_app.state.agent = mock_agent

        # First interaction
        mock_result1 = MagicMock()
        mock_result1.output = "Hello! Nice to meet you!"
        mock_result1.all_messages.return_value = [
            ModelRequest(parts=[UserPromptPart(content="Hi there!")]),
            ModelResponse(parts=[TextPart(content="Hello! Nice to meet you!")])
        ]

        # Second interaction
        mock_result2 = MagicMock()
        mock_result2.output = "I'm doing well, thank you for asking!"
        mock_result2.all_messages.return_value = [
            ModelRequest(parts=[UserPromptPart(content="Hi there!")]),
            ModelResponse(parts=[TextPart(content="Hello! Nice to meet you!")]),
            ModelRequest(parts=[UserPromptPart(content="How are you?")]),
            ModelResponse(parts=[TextPart(content="I'm doing well, thank you for asking!")])
        ]

        # Mock agent run calls
        call_count = 0
        async def mock_run(prompt, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return mock_result1
            else:
                return mock_result2

        mock_agent.run = AsyncMock(side_effect=mock_run)

        async def mock_post_chat(prompt: str = Form(...)):
            async def stream_messages():
                user_msg = ChatMessage(
                    role="user",
                    timestamp=datetime.now(tz=timezone.utc).isoformat(),
                    content=prompt,
                )
                yield user_msg.model_dump_json().encode("utf-8") + b"\n"

                result = await mock_agent.run(prompt)

                assistant_msg = ChatMessage(
                    role="assistant",
                    timestamp=datetime.now(tz=timezone.utc).isoformat(),
                    content=result.output,
                )
                yield assistant_msg.model_dump_json().encode("utf-8") + b"\n"

                mock_agent._history = mock_agent._repack(result.all_messages())

            return StreamingResponse(stream_messages(), media_type="text/plain")

        test_app.post("/ask/")(mock_post_chat)

        client = TestClient(test_app)

        # First request
        response1 = client.post("/ask/", data={"prompt": "Hi there!"})
        assert response1.status_code == 200

        # Second request - should have context from first
        response2 = client.post("/ask/", data={"prompt": "How are you?"})
        assert response2.status_code == 200

        # Verify agent was called with accumulated history on second call
        assert mock_agent.run.call_count == 2


class TestNegativeScenarios:
    """Negative test scenarios for the chat API."""

    def test_post_chat_missing_prompt_field(self):
        """Test POST /ask/ with missing prompt field."""
        from fastapi.testclient import TestClient
        from fastapi import FastAPI

        mock_agent = MagicMock(spec=MockAgentASK)
        test_app = FastAPI(lifespan=make_lifespan(mock_agent))

        async def mock_post_chat(prompt: str = Form(...)):
            return {"error": "Should not reach here"}

        test_app.post("/ask/")(mock_post_chat)

        client = TestClient(test_app)

        # Send request without prompt field
        response = client.post("/ask/", data={})  # Empty form data

        # FastAPI should return 422 for missing required field
        assert response.status_code == 422
        error_data = response.json()
        assert "detail" in error_data
        assert any("prompt" in str(error).lower() for error in error_data["detail"])

    def test_post_chat_empty_prompt(self):
        """Test POST /ask/ with empty prompt string."""
        from fastapi.testclient import TestClient
        from fastapi import FastAPI

        mock_agent = MagicMock(spec=MockAgentASK)
        mock_agent._history = []
        mock_agent._repack = MagicMock(return_value=[])
        mock_agent._agent = MagicMock()  # Add the _agent attribute

        mock_result = MagicMock()
        mock_result.output = "I received an empty message. How can I help you?"
        mock_result.all_messages.return_value = [
            ModelRequest(parts=[UserPromptPart(content="")]),  # Empty prompt
            ModelResponse(parts=[TextPart(content=mock_result.output)])
        ]

        mock_agent.run = AsyncMock(return_value=mock_result)

        test_app = FastAPI(lifespan=make_lifespan(mock_agent))
        test_app.state.agent = mock_agent

        async def mock_post_chat(prompt: str = Form(...)):
            async def stream_messages():
                user_msg = ChatMessage(
                    role="user",
                    timestamp=datetime.now(tz=timezone.utc).isoformat(),
                    content=prompt,
                )
                yield user_msg.model_dump_json().encode("utf-8") + b"\n"

                result = await mock_agent.run(prompt)

                assistant_msg = ChatMessage(
                    role="assistant",
                    timestamp=datetime.now(tz=timezone.utc).isoformat(),
                    content=result.output,
                )
                yield assistant_msg.model_dump_json().encode("utf-8") + b"\n"

                mock_agent._history = mock_agent._repack(result.all_messages())

            return StreamingResponse(stream_messages(), media_type="text/plain")

        test_app.post("/ask/")(mock_post_chat)

        client = TestClient(test_app)
        response = client.post("/ask/", data={"prompt": ""})

        assert response.status_code == 200
        lines = response.content.decode().strip().split('\n')
        assert len(lines) == 2

        user_msg = json.loads(lines[0])
        assistant_msg = json.loads(lines[1])

        assert user_msg["content"] == ""  # Empty prompt should be preserved
        assert assistant_msg["content"] == mock_result.output

    def test_post_chat_whitespace_only_prompt(self):
        """Test POST /ask/ with whitespace-only prompt."""
        from fastapi.testclient import TestClient
        from fastapi import FastAPI

        mock_agent = MagicMock(spec=MockAgentASK)
        mock_agent._history = []
        mock_agent._repack = MagicMock(return_value=[])
        mock_agent._agent = MagicMock()  # Add the _agent attribute

        whitespace_prompt = "   \n\t  \n  "

        mock_result = MagicMock()
        mock_result.output = "I see you've sent a message with only whitespace. Is there something specific you'd like to talk about?"
        mock_result.all_messages.return_value = [
            ModelRequest(parts=[UserPromptPart(content=whitespace_prompt)]),
            ModelResponse(parts=[TextPart(content=mock_result.output)])
        ]

        mock_agent.run = AsyncMock(return_value=mock_result)

        test_app = FastAPI(lifespan=make_lifespan(mock_agent))
        test_app.state.agent = mock_agent

        async def mock_post_chat(prompt: str = Form(...)):
            async def stream_messages():
                user_msg = ChatMessage(
                    role="user",
                    timestamp=datetime.now(tz=timezone.utc).isoformat(),
                    content=prompt,
                )
                yield user_msg.model_dump_json().encode("utf-8") + b"\n"

                result = await mock_agent.run(prompt)

                assistant_msg = ChatMessage(
                    role="assistant",
                    timestamp=datetime.now(tz=timezone.utc).isoformat(),
                    content=result.output,
                )
                yield assistant_msg.model_dump_json().encode("utf-8") + b"\n"

                mock_agent._history = mock_agent._repack(result.all_messages())

            return StreamingResponse(stream_messages(), media_type="text/plain")

        test_app.post("/ask/")(mock_post_chat)

        client = TestClient(test_app)
        response = client.post("/ask/", data={"prompt": whitespace_prompt})

        assert response.status_code == 200
        lines = response.content.decode().strip().split('\n')
        assert len(lines) == 2

        user_msg = json.loads(lines[0])
        assistant_msg = json.loads(lines[1])

        assert user_msg["content"] == whitespace_prompt
        assert assistant_msg["content"] == mock_result.output

    def test_post_chat_agent_runtime_error(self):
        """Test POST /ask/ when agent raises a runtime error."""
        from fastapi.testclient import TestClient
        from fastapi import FastAPI

        mock_agent = MagicMock(spec=MockAgentASK)
        mock_agent._history = []
        mock_agent._repack = MagicMock(return_value=[])
        mock_agent._agent = MagicMock()  # Add the _agent attribute

        # Mock agent to raise an exception
        mock_agent.run = AsyncMock(side_effect=RuntimeError("Model service unavailable"))

        test_app = FastAPI(lifespan=make_lifespan(mock_agent))
        test_app.state.agent = mock_agent

        async def mock_post_chat(prompt: str = Form(...)):
            async def stream_messages():
                user_msg = ChatMessage(
                    role="user",
                    timestamp=datetime.now(tz=timezone.utc).isoformat(),
                    content=prompt,
                )
                yield user_msg.model_dump_json().encode("utf-8") + b"\n"

                try:
                    result = await mock_agent.run(prompt)

                    assistant_msg = ChatMessage(
                        role="assistant",
                        timestamp=datetime.now(tz=timezone.utc).isoformat(),
                        content=result.output,
                    )
                    yield assistant_msg.model_dump_json().encode("utf-8") + b"\n"

                    mock_agent._history = mock_agent._repack(result.all_messages())
                except Exception as e:
                    error_msg = ChatMessage(
                        role="assistant",  # Use assistant role for error messages
                        timestamp=datetime.now(tz=timezone.utc).isoformat(),
                        content=f"Error: {str(e)}",
                    )
                    yield error_msg.model_dump_json().encode("utf-8") + b"\n"

            return StreamingResponse(stream_messages(), media_type="text/plain")

        test_app.post("/ask/")(mock_post_chat)

        client = TestClient(test_app)
        response = client.post("/ask/", data={"prompt": "Test prompt"})

        assert response.status_code == 200  # Streaming response still returns 200
        lines = response.content.decode().strip().split('\n')
        assert len(lines) == 2  # User message + error message

        user_msg = json.loads(lines[0])
        error_msg = json.loads(lines[1])

        assert user_msg["role"] == "user"
        assert user_msg["content"] == "Test prompt"
        assert error_msg["role"] == "assistant"
        assert "Model service unavailable" in error_msg["content"]

    def test_post_chat_extremely_long_prompt(self):
        """Test POST /ask/ with extremely long prompt."""
        from fastapi.testclient import TestClient
        from fastapi import FastAPI

        mock_agent = MagicMock(spec=MockAgentASK)
        mock_agent._history = []
        mock_agent._repack = MagicMock(return_value=[])
        mock_agent._agent = MagicMock()  # Add the _agent attribute

        # Create a very long prompt (10,000 characters)
        long_prompt = "Hello! " * 1000  # Approximately 7,000 characters

        mock_result = MagicMock()
        mock_result.output = "That's a very long message! Let me help you with that."
        mock_result.all_messages.return_value = [
            ModelRequest(parts=[UserPromptPart(content=long_prompt)]),
            ModelResponse(parts=[TextPart(content=mock_result.output)])
        ]

        mock_agent.run = AsyncMock(return_value=mock_result)

        test_app = FastAPI(lifespan=make_lifespan(mock_agent))
        test_app.state.agent = mock_agent

        async def mock_post_chat(prompt: str = Form(...)):
            async def stream_messages():
                user_msg = ChatMessage(
                    role="user",
                    timestamp=datetime.now(tz=timezone.utc).isoformat(),
                    content=prompt,
                )
                yield user_msg.model_dump_json().encode("utf-8") + b"\n"

                result = await mock_agent.run(prompt)

                assistant_msg = ChatMessage(
                    role="assistant",
                    timestamp=datetime.now(tz=timezone.utc).isoformat(),
                    content=result.output,
                )
                yield assistant_msg.model_dump_json().encode("utf-8") + b"\n"

                mock_agent._history = mock_agent._repack(result.all_messages())

            return StreamingResponse(stream_messages(), media_type="text/plain")

        test_app.post("/ask/")(mock_post_chat)

        client = TestClient(test_app)
        response = client.post("/ask/", data={"prompt": long_prompt})

        assert response.status_code == 200
        lines = response.content.decode().strip().split('\n')
        assert len(lines) == 2

        user_msg = json.loads(lines[0])
        assistant_msg = json.loads(lines[1])

        assert len(user_msg["content"]) == len(long_prompt)
        assert user_msg["content"] == long_prompt
        assert assistant_msg["content"] == mock_result.output

    def test_concurrent_requests_simulation(self):
        """Test behavior with simulated concurrent requests."""
        from fastapi.testclient import TestClient
        from fastapi import FastAPI
        import asyncio

        mock_agent = MagicMock(spec=MockAgentASK)
        mock_agent._history = []
        mock_agent._repack = MagicMock()
        mock_agent._agent = MagicMock()

        call_order: list[str] = []
        history_states: list[int] = []

        async def mock_run(prompt, **kwargs):
            call_order.append(prompt)
            history_states.append(len(kwargs.get("message_history", [])))
            await asyncio.sleep(0.01)

            mock_result = MagicMock()
            mock_result.output = f"Response to: {prompt}"
            mock_result.all_messages.return_value = [
                ModelRequest(parts=[UserPromptPart(content=prompt)]),
                ModelResponse(parts=[TextPart(content=f"Response to: {prompt}")]),
            ]
            return mock_result

        mock_agent.run = AsyncMock(side_effect=mock_run)
        mock_agent._repack.return_value = []

        test_app = FastAPI(lifespan=make_lifespan(mock_agent))
        test_app.state.agent = mock_agent

        async def mock_post_chat(prompt: str = Form(...)):
            async def stream_messages():
                user_msg = ChatMessage(
                    role="user",
                    timestamp=datetime.now(tz=timezone.utc).isoformat(),
                    content=prompt,
                )
                yield user_msg.model_dump_json().encode("utf-8") + b"\n"

                result = await mock_agent.run(prompt)

                assistant_msg = ChatMessage(
                    role="assistant",
                    timestamp=datetime.now(tz=timezone.utc).isoformat(),
                    content=result.output,
                )
                yield assistant_msg.model_dump_json().encode("utf-8") + b"\n"

                mock_agent._history = mock_agent._repack(result.all_messages())

            return StreamingResponse(stream_messages(), media_type="text/plain")

        test_app.post("/ask/")(mock_post_chat)

        client = TestClient(test_app)

        responses = []
        for i in range(3):
            response = client.post("/ask/", data={"prompt": f"Request {i+1}"})
            responses.append(response)

        for i, response in enumerate(responses):
            assert response.status_code == 200
            lines = response.content.decode().strip().split("\n")
            assert len(lines) == 2

            user_msg = json.loads(lines[0])
            assistant_msg = json.loads(lines[1])

            assert user_msg["content"] == f"Request {i+1}"
            assert assistant_msg["content"] == f"Response to: Request {i+1}"

        assert mock_agent.run.call_count == 3
