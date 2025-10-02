import atexit
import sys

from langfuse import get_client
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor
from pydantic_ai import Agent

from ask.core.config import TraceConfig


def setup_instrumentation_config(
    config: TraceConfig,
) -> None:
    """Sets up OpenTelemetry instrumentation with Langfuse."""
    import os

    os.environ["LANGFUSE_PUBLIC_KEY"] = config.public_key
    os.environ["LANGFUSE_SECRET_KEY"] = config.secret_key
    os.environ["LANGFUSE_HOST"] = config.host_url
    get_client()  # Initialize the Langfuse client
    Agent.instrument_all()
    print(f"Telemetry logging to Langfuse: {config.host_url}", file=sys.stderr)


def setup_instrumentation_file(
    file: str,
) -> None:
    """Sets up OpenTelemetry instrumentation to log traces to a file."""
    provider = TracerProvider()
    file_stream = open(file, "w", encoding="utf-8")
    provider.add_span_processor(
        SimpleSpanProcessor(ConsoleSpanExporter(out=file_stream))
    )
    trace.set_tracer_provider(provider)
    atexit.register(file_stream.close)
    print(f"Telemetry logging to: {file}", file=sys.stderr)
