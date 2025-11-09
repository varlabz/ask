from datetime import datetime
from typing import Any

import pytest
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models import Model
from pydantic_ai.usage import RequestUsage

from ask.core.model_wrapper import ThoughtWrapper


class MockModel(Model):
    def __init__(self, response: ModelResponse):
        self._response = response

    async def request(self, *args: Any, **kwargs: Any) -> ModelResponse:
        return self._response

    @property
    def model_name(self) -> str:
        return "mock_model"

    @property
    def system(self) -> str:
        return ""


@pytest.mark.asyncio
async def test_thought_wrapper_modifies_response():
    """Positive test: on_before_tool_use returns a new string."""
    original_content = "Original thought"
    modified_content = "Modified thought"

    def on_before_tool_use(thought: str) -> str | None:
        assert thought == original_content
        return modified_content

    tool_call = ToolCallPart(tool_name="test_tool", args={}, tool_call_id="123")
    response = ModelResponse(
        parts=[TextPart(content=original_content), tool_call],
        usage=RequestUsage(input_tokens=1, output_tokens=1),
        model_name="test_model",
        timestamp=datetime.now(),
    )

    model = MockModel(response)
    wrapper = ThoughtWrapper(model, on_before_tool_use)

    result = await wrapper.request()

    assert len(result.parts) == 2
    assert isinstance(result.parts[0], TextPart)
    assert result.parts[0].content == modified_content
    assert result.parts[1] == tool_call


@pytest.mark.asyncio
async def test_thought_wrapper_no_modification():
    """Negative test: on_before_tool_use returns None."""
    original_content = "Original thought"

    def on_before_tool_use(thought: str) -> str | None:
        assert thought == original_content
        return None

    tool_call = ToolCallPart(tool_name="test_tool", args={}, tool_call_id="123")
    response = ModelResponse(
        parts=[TextPart(content=original_content), tool_call],
        usage=RequestUsage(input_tokens=1, output_tokens=1),
        model_name="test_model",
        timestamp=datetime.now(),
    )

    model = MockModel(response)
    wrapper = ThoughtWrapper(model, on_before_tool_use)

    result = await wrapper.request()

    assert result == response
    assert result.parts[0].content == original_content


@pytest.mark.asyncio
async def test_thought_wrapper_no_tool_calls():
    """Test case where there are no tool calls in the response."""
    original_content = "Original thought"
    called = False

    def on_before_tool_use(thought: str) -> str | None:
        nonlocal called
        called = True
        return None

    response = ModelResponse(
        parts=[TextPart(content=original_content)],
        usage=RequestUsage(input_tokens=1, output_tokens=1),
        model_name="test_model",
        timestamp=datetime.now(),
    )

    model = MockModel(response)
    wrapper = ThoughtWrapper(model, on_before_tool_use)

    result = await wrapper.request()

    assert not called
    assert result == response
