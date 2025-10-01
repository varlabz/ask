import json
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import (
    SimpleSpanProcessor,
    SpanExporter,
    SpanExportResult,
)
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)


class CallbackSpanExporter(SpanExporter):
    """SpanExporter that sends each span to a callback function."""

    def __init__(self, callback: Callable[[ReadableSpan], None]):
        self.callback = callback

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        """Export spans by calling the callback for each span."""
        for span in spans:
            try:
                self.callback(span)
            except Exception as e:
                print(f"Error in span callback: {e}", file=sys.stderr)
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        """Shutdown the exporter."""
        pass


def setup_instrumentation_with_callback(
    callback: Callable[[ReadableSpan], None],
    service_name: str = "pydantic-ai-agent",
) -> None:
    """
    Set up OpenTelemetry instrumentation with a callback to receive span data.

    Args:
        callback: Function called with each ReadableSpan
        service_name: Service name for telemetry
    """
    resource = Resource(attributes={"service.name": service_name})
    provider = TracerProvider(resource=resource)

    exporter = CallbackSpanExporter(callback)
    processor = SimpleSpanProcessor(exporter)
    provider.add_span_processor(processor)

    trace.set_tracer_provider(provider)

    print(f"Instrumentation enabled for '{service_name}'", file=sys.stderr)


@dataclass
class LLMCallData:
    """Structured data extracted from an LLM call span."""

    span_name: str
    prompt: Any | None = None
    tools: Any | None = None
    result: Any | None = None
    model: str | None = None
    usage: dict[str, int] | None = None
    duration_ms: float | None = None
    status: str | None = None
    timestamp: int | None = None
    model_request_parameters: dict[str, Any] | None = None
    input_messages: list[dict[str, Any]] | None = None
    output_messages: list[dict[str, Any]] | None = None
    # Parsed MCP message structures
    parsed_input: list[ModelMessage] | None = field(default=None)
    parsed_output: list[ModelMessage] | None = field(default=None)


def extract_llm_call_data(span: ReadableSpan) -> LLMCallData | None:
    """
    Extract LLM call information from a ReadableSpan.

    Returns LLMCallData with parsed message structures and metadata,
    or None if the span is not LLM-related.
    """
    if not span.attributes:
        return None

    # Check if this is an LLM-related span
    span_name = span.name.lower()
    if not any(
        keyword in span_name for keyword in ["llm", "model", "request", "call", "chat"]
    ):
        return None

    # Extract relevant attributes
    attrs = span.attributes

    # Extract and convert usage tokens to integers
    usage_data = {
        "input_tokens": attrs.get("gen_ai.usage.input_tokens"),
        "output_tokens": attrs.get("gen_ai.usage.output_tokens"),
        "total_tokens": attrs.get("gen_ai.usage.total_tokens"),
    }

    usage = {}
    for key, value in usage_data.items():
        if value is not None:
            try:
                usage[key] = int(str(value))
            except (ValueError, TypeError):
                pass

    usage = usage or None

    # Extract model name as string
    model = attrs.get("gen_ai.request.model") or attrs.get("gen_ai.system")
    model_str = str(model) if model is not None else None

    # Helper: Parse JSON strings to Python objects
    def parse_json(value: Any) -> Any:
        """Parse JSON string, return original value if parsing fails."""
        if isinstance(value, str):
            try:
                return json.loads(value)
            except (json.JSONDecodeError, TypeError):
                return value
        return value

    # Helper: Parse messages into pydantic-ai structures
    def parse_messages(messages_data: Any) -> list[ModelMessage] | None:
        """Convert raw message data to typed pydantic-ai ModelMessage objects."""
        if not messages_data or not isinstance(messages_data, list):
            return None

        parsed = []

        for msg in messages_data:
            if not isinstance(msg, dict):
                continue

            role = msg.get("role")
            parts = msg.get("parts", [])
            parsed_parts = []

            for part in parts:
                if not isinstance(part, dict):
                    continue

                part_type = part.get("type")

                # Common parts for both user and assistant
                if part_type == "text":
                    parsed_parts.append(TextPart(content=part.get("content", "")))
                elif part_type == "thinking":
                    parsed_parts.append(ThinkingPart(content=part.get("content", "")))
                elif part_type == "user-prompt":
                    parsed_parts.append(UserPromptPart(content=part.get("content", "")))

                # Tool-related parts
                elif part_type == "tool_call":
                    args_raw = part.get("arguments", "{}")
                    args = (
                        json.loads(args_raw)
                        if isinstance(args_raw, str)
                        else (args_raw if isinstance(args_raw, dict) else {})
                    )
                    parsed_parts.append(
                        ToolCallPart(
                            tool_name=part.get("name", ""),
                            args=args,
                            tool_call_id=str(part.get("id", "")),
                        )
                    )
                elif part_type == "tool_call_response":
                    parsed_parts.append(
                        ToolReturnPart(
                            tool_name=part.get("name", ""),
                            content=part.get("result", ""),
                            tool_call_id=str(part.get("id", "")),
                        )
                    )

            # Add message with parsed parts
            if parsed_parts:
                if role == "user":
                    parsed.append(ModelRequest(parts=parsed_parts))
                elif role == "assistant":
                    parsed.append(ModelResponse(parts=parsed_parts))

        return parsed or None

    # Parse JSON attributes
    model_request_params = parse_json(attrs.get("model_request_parameters"))
    input_msgs = parse_json(attrs.get("gen_ai.input.messages"))
    output_msgs = parse_json(attrs.get("gen_ai.output.messages"))

    # Convert to typed message structures
    parsed_input = parse_messages(input_msgs) if input_msgs else None
    parsed_output = parse_messages(output_msgs) if output_msgs else None

    return LLMCallData(
        span_name=span.name,
        prompt=attrs.get("gen_ai.prompt") or attrs.get("gen_ai.request.messages"),
        tools=attrs.get("gen_ai.tools") or attrs.get("gen_ai.request.tools"),
        result=attrs.get("gen_ai.response") or attrs.get("gen_ai.response.text"),
        model=model_str,
        usage=usage,
        duration_ms=(span.end_time - span.start_time) / 1_000_000
        if span.end_time and span.start_time
        else None,
        status=span.status.status_code.name if span.status else None,
        timestamp=span.start_time,
        model_request_parameters=model_request_params,
        input_messages=input_msgs,
        output_messages=output_msgs,
        parsed_input=parsed_input,
        parsed_output=parsed_output,
    )
