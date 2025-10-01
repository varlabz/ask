"""
Tests for instrumentation module with callback-based span exporting and LLM call data extraction.
"""

import json
from unittest.mock import Mock

import pytest
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.trace.status import StatusCode
from pydantic_ai.messages import (
    ModelResponse,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
)

from ask.eval.instrumentation import (
    CallbackSpanExporter,
    LLMCallData,
    extract_llm_call_data,
    setup_instrumentation_with_callback,
)


class TestCallbackSpanExporter:
    """Tests for CallbackSpanExporter class."""

    def test_callback_span_exporter_success(self):
        """Test that callback is called for each span successfully."""
        callback_calls = []

        def callback(span):
            callback_calls.append(span)

        exporter = CallbackSpanExporter(callback)

        mock_span = Mock(spec=ReadableSpan)
        spans = [mock_span, mock_span, mock_span]

        result = exporter.export(spans)

        assert len(callback_calls) == 3
        assert result.name == "SUCCESS"

    def test_callback_span_exporter_handles_error(self):
        """Test that exporter handles callback errors gracefully."""

        def failing_callback(span):
            raise ValueError("Callback error")

        exporter = CallbackSpanExporter(failing_callback)
        mock_span = Mock(spec=ReadableSpan)

        # Should not raise, but handle the error
        result = exporter.export([mock_span])
        assert result.name == "SUCCESS"

    def test_callback_span_exporter_shutdown(self):
        """Test that shutdown method exists and can be called."""
        callback = Mock()
        exporter = CallbackSpanExporter(callback)
        exporter.shutdown()  # Should not raise


class TestSetupInstrumentationWithCallback:
    """Tests for setup_instrumentation_with_callback function."""

    def test_setup_instrumentation_with_callback(self):
        """Test that instrumentation setup completes without errors."""
        callback = Mock()
        # Should not raise
        setup_instrumentation_with_callback(callback, service_name="test-service")

    def test_setup_with_custom_service_name(self):
        """Test setup with custom service name."""
        callback = Mock()
        setup_instrumentation_with_callback(
            callback, service_name="custom-agent-service"
        )


class TestExtractLLMCallDataPositive:
    """Positive tests for extract_llm_call_data function."""

    def create_mock_span(
        self,
        span_name="chat x-ai/grok-code-fast-1",
        attributes=None,
        start_time=1000000000,
        end_time=2000000000,
    ):
        """Helper to create a mock ReadableSpan."""
        mock_span = Mock(spec=ReadableSpan)
        mock_span.name = span_name
        mock_span.attributes = attributes or {}
        mock_span.start_time = start_time
        mock_span.end_time = end_time

        # Create a mock status with proper status_code
        mock_status = Mock()
        mock_status.status_code = StatusCode.UNSET
        mock_span.status = mock_status
        return mock_span

    def test_extract_basic_llm_call_data(self):
        """Test extraction of basic LLM call data."""
        attributes = {
            "gen_ai.request.model": "gpt-4",
            "gen_ai.system": "openai",
            "gen_ai.usage.input_tokens": 100,
            "gen_ai.usage.output_tokens": 50,
            "gen_ai.usage.total_tokens": 150,
        }
        span = self.create_mock_span(attributes=attributes)

        result = extract_llm_call_data(span)

        assert result is not None
        assert isinstance(result, LLMCallData)
        assert result.span_name == "chat x-ai/grok-code-fast-1"
        assert result.model == "gpt-4"
        assert result.usage == {
            "input_tokens": 100,
            "output_tokens": 50,
            "total_tokens": 150,
        }
        assert result.duration_ms == 1000.0  # (2000000000 - 1000000000) / 1_000_000

    def test_extract_with_json_messages(self):
        """Test extraction with JSON message strings."""
        input_messages = [
            {
                "role": "user",
                "parts": [{"type": "text", "content": "Hello, world!"}],
            }
        ]
        output_messages = [
            {
                "role": "assistant",
                "parts": [{"type": "text", "content": "Hi there!"}],
            }
        ]

        attributes = {
            "gen_ai.request.model": "gpt-4",
            "gen_ai.input.messages": json.dumps(input_messages),
            "gen_ai.output.messages": json.dumps(output_messages),
            "model_request_parameters": json.dumps(
                {"temperature": 0.7, "output_mode": "text"}
            ),
        }
        span = self.create_mock_span(attributes=attributes)

        result = extract_llm_call_data(span)

        assert result is not None
        assert result.input_messages == input_messages
        assert result.output_messages == output_messages
        assert result.model_request_parameters == {
            "temperature": 0.7,
            "output_mode": "text",
        }

    def test_extract_with_tool_calls(self):
        """Test extraction with tool call messages."""
        output_messages = [
            {
                "role": "assistant",
                "parts": [
                    {
                        "type": "tool_call",
                        "name": "search",
                        "id": "call_123",
                        "arguments": json.dumps({"query": "test"}),
                    }
                ],
            }
        ]

        attributes = {
            "gen_ai.request.model": "gpt-4",
            "gen_ai.output.messages": json.dumps(output_messages),
        }
        span = self.create_mock_span(attributes=attributes)

        result = extract_llm_call_data(span)

        assert result is not None
        assert result.parsed_output is not None
        assert len(result.parsed_output) == 1
        assert isinstance(result.parsed_output[0], ModelResponse)

        tool_call_found = False
        for part in result.parsed_output[0].parts:
            if isinstance(part, ToolCallPart):
                tool_call_found = True
                assert part.tool_name == "search"
                assert part.tool_call_id == "call_123"
                assert part.args == {"query": "test"}

        assert tool_call_found, "ToolCallPart not found in parsed output"

    def test_extract_with_thinking_part(self):
        """Test extraction with thinking/reasoning content."""
        output_messages = [
            {
                "role": "assistant",
                "parts": [
                    {
                        "type": "thinking",
                        "content": "Let me think about this problem...",
                    },
                    {"type": "text", "content": "Here is my answer."},
                ],
            }
        ]

        attributes = {
            "gen_ai.request.model": "gpt-4",
            "gen_ai.output.messages": json.dumps(output_messages),
        }
        span = self.create_mock_span(attributes=attributes)

        result = extract_llm_call_data(span)

        assert result is not None
        assert result.parsed_output is not None
        assert len(result.parsed_output) == 1

        thinking_found = False
        text_found = False
        for part in result.parsed_output[0].parts:
            if isinstance(part, ThinkingPart):
                thinking_found = True
                assert "think about this problem" in part.content
            elif isinstance(part, TextPart):
                text_found = True
                assert part.content == "Here is my answer."

        assert thinking_found and text_found

    def test_extract_with_tool_return(self):
        """Test extraction with tool return/response."""
        input_messages = [
            {
                "role": "user",
                "parts": [
                    {
                        "type": "tool_call_response",
                        "name": "search",
                        "id": "call_123",
                        "result": "Search results here",
                    }
                ],
            }
        ]

        attributes = {
            "gen_ai.request.model": "gpt-4",
            "gen_ai.input.messages": json.dumps(input_messages),
        }
        span = self.create_mock_span(attributes=attributes)

        result = extract_llm_call_data(span)

        assert result is not None
        assert result.parsed_input is not None
        assert len(result.parsed_input) == 1

        tool_return_found = False
        for part in result.parsed_input[0].parts:
            if isinstance(part, ToolReturnPart):
                tool_return_found = True
                assert part.tool_name == "search"
                assert part.tool_call_id == "call_123"
                assert "Search results" in str(part.content)

        assert tool_return_found

    def test_extract_with_model_request_parameters(self):
        """Test extraction of model request parameters."""
        params = {
            "function_tools": [
                {"name": "search", "description": "Search the web"},
                {"name": "calculator", "description": "Do math"},
            ],
            "output_mode": "text",
            "temperature": 0.0,
            "builtin_tools": [],
        }

        attributes = {
            "gen_ai.request.model": "gpt-4",
            "model_request_parameters": json.dumps(params),
        }
        span = self.create_mock_span(attributes=attributes)

        result = extract_llm_call_data(span)

        assert result is not None
        assert result.model_request_parameters is not None
        assert len(result.model_request_parameters["function_tools"]) == 2
        assert result.model_request_parameters["temperature"] == 0.0
        assert result.model_request_parameters["output_mode"] == "text"


class TestExtractLLMCallDataNegative:
    """Negative tests for extract_llm_call_data function."""

    def create_mock_span(
        self, span_name="some span", attributes=None, start_time=None, end_time=None
    ):
        """Helper to create a mock ReadableSpan."""
        mock_span = Mock(spec=ReadableSpan)
        mock_span.name = span_name
        mock_span.attributes = attributes
        mock_span.start_time = start_time
        mock_span.end_time = end_time

        # Create a mock status with proper status_code
        if attributes:
            mock_status = Mock()
            mock_status.status_code = StatusCode.UNSET
            mock_span.status = mock_status
        else:
            mock_span.status = None
        return mock_span

    def test_extract_returns_none_for_no_attributes(self):
        """Test that extraction returns None when span has no attributes."""
        span = self.create_mock_span(attributes=None)
        result = extract_llm_call_data(span)
        assert result is None

    def test_extract_returns_none_for_non_llm_span(self):
        """Test that extraction returns None for non-LLM related spans."""
        attributes = {"some.other.attribute": "value"}
        span = self.create_mock_span(span_name="database query", attributes=attributes)
        result = extract_llm_call_data(span)
        assert result is None

    def test_extract_handles_invalid_json_messages(self):
        """Test that extraction handles invalid JSON gracefully."""
        attributes = {
            "gen_ai.request.model": "gpt-4",
            "gen_ai.input.messages": "not valid json {{{",
            "gen_ai.output.messages": "also not valid",
        }
        span = self.create_mock_span(
            span_name="chat model", attributes=attributes, start_time=0, end_time=1000
        )

        result = extract_llm_call_data(span)

        assert result is not None
        # Should keep the original strings when JSON parsing fails
        assert result.input_messages == "not valid json {{{"
        assert result.output_messages == "also not valid"

    def test_extract_handles_missing_usage_tokens(self):
        """Test extraction when usage token data is missing."""
        attributes = {
            "gen_ai.request.model": "gpt-4",
        }
        span = self.create_mock_span(
            span_name="chat model", attributes=attributes, start_time=0, end_time=1000
        )

        result = extract_llm_call_data(span)

        assert result is not None
        assert result.usage is None

    def test_extract_handles_partial_usage_tokens(self):
        """Test extraction with partial usage token data."""
        attributes = {
            "gen_ai.request.model": "gpt-4",
            "gen_ai.usage.input_tokens": 100,
            # Missing output_tokens and total_tokens
        }
        span = self.create_mock_span(
            span_name="chat model", attributes=attributes, start_time=0, end_time=1000
        )

        result = extract_llm_call_data(span)

        assert result is not None
        assert result.usage is not None
        assert result.usage["input_tokens"] == 100
        assert "output_tokens" not in result.usage

    def test_extract_handles_invalid_token_types(self):
        """Test extraction with invalid token types."""
        attributes = {
            "gen_ai.request.model": "gpt-4",
            "gen_ai.usage.input_tokens": "not a number",
            "gen_ai.usage.output_tokens": None,
        }
        span = self.create_mock_span(
            span_name="chat model", attributes=attributes, start_time=0, end_time=1000
        )

        result = extract_llm_call_data(span)

        assert result is not None
        # Should handle conversion errors gracefully

    def test_extract_handles_empty_messages_array(self):
        """Test extraction with empty messages arrays."""
        attributes = {
            "gen_ai.request.model": "gpt-4",
            "gen_ai.input.messages": json.dumps([]),
            "gen_ai.output.messages": json.dumps([]),
        }
        span = self.create_mock_span(
            span_name="chat model", attributes=attributes, start_time=0, end_time=1000
        )

        result = extract_llm_call_data(span)

        assert result is not None
        assert result.input_messages == []
        assert result.output_messages == []
        assert result.parsed_input is None
        assert result.parsed_output is None

    def test_extract_handles_malformed_message_structure(self):
        """Test extraction with malformed message structure."""
        malformed_messages = [
            {
                "role": "user",
                # Missing 'parts' field
            },
            {
                # Missing 'role' field
                "parts": [{"type": "text", "content": "test"}]
            },
        ]

        attributes = {
            "gen_ai.request.model": "gpt-4",
            "gen_ai.input.messages": json.dumps(malformed_messages),
        }
        span = self.create_mock_span(
            span_name="chat model", attributes=attributes, start_time=0, end_time=1000
        )

        result = extract_llm_call_data(span)

        assert result is not None
        # Should handle malformed messages gracefully without crashing

    def test_extract_handles_missing_tool_call_fields(self):
        """Test extraction with tool call missing required fields."""
        output_messages = [
            {
                "role": "assistant",
                "parts": [
                    {
                        "type": "tool_call",
                        # Missing 'name', 'id', 'arguments'
                    }
                ],
            }
        ]

        attributes = {
            "gen_ai.request.model": "gpt-4",
            "gen_ai.output.messages": json.dumps(output_messages),
        }
        span = self.create_mock_span(
            span_name="chat model", attributes=attributes, start_time=0, end_time=1000
        )

        result = extract_llm_call_data(span)

        assert result is not None
        # Should handle missing fields gracefully

    def test_extract_handles_none_timestamps(self):
        """Test extraction when timestamps are None."""
        attributes = {
            "gen_ai.request.model": "gpt-4",
        }
        span = self.create_mock_span(
            span_name="chat model",
            attributes=attributes,
            start_time=None,
            end_time=None,
        )

        result = extract_llm_call_data(span)

        assert result is not None
        assert result.duration_ms is None

    def test_extract_handles_non_string_model_name(self):
        """Test extraction with non-string model name."""
        attributes = {
            "gen_ai.request.model": 12345,  # Non-string value
        }
        span = self.create_mock_span(
            span_name="chat model", attributes=attributes, start_time=0, end_time=1000
        )

        result = extract_llm_call_data(span)

        assert result is not None
        assert result.model == "12345"  # Should be converted to string


class TestLLMCallDataStructure:
    """Tests for LLMCallData dataclass structure."""

    def test_llm_call_data_creation(self):
        """Test creating LLMCallData instance."""
        data = LLMCallData(
            span_name="test span",
            model="gpt-4",
            usage={"input_tokens": 10, "output_tokens": 5},
            duration_ms=123.45,
        )

        assert data.span_name == "test span"
        assert data.model == "gpt-4"
        assert data.usage is not None
        assert data.usage["input_tokens"] == 10
        assert data.duration_ms == 123.45

    def test_llm_call_data_defaults(self):
        """Test LLMCallData default values."""
        data = LLMCallData(span_name="test")

        assert data.span_name == "test"
        assert data.prompt is None
        assert data.tools is None
        assert data.result is None
        assert data.model is None
        assert data.usage is None
        assert data.duration_ms is None
        assert data.status is None
        assert data.timestamp is None
        assert data.model_request_parameters is None
        assert data.input_messages is None
        assert data.output_messages is None
        assert data.parsed_input is None
        assert data.parsed_output is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
