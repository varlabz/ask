import atexit
import sys

from langfuse import Langfuse
from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor

from ask.core.config import TraceConfig


def _create_tracer_provider(service_name: str = "ask-agent") -> TracerProvider:
    """Creates a TracerProvider with the given service name."""
    resource = Resource(attributes={"service.name": service_name})
    return TracerProvider(resource=resource)


def setup_instrumentation_config(
    config: TraceConfig,
) -> None:
    """Sets up OpenTelemetry instrumentation with Langfuse."""
    provider = _create_tracer_provider()
    client = Langfuse(
        public_key=config.public_key,
        secret_key=config.secret_key,
        host=config.host_url,
        tracer_provider=provider,
    )
    trace.set_tracer_provider(provider)
    atexit.register(client.shutdown)
    print(f"Telemetry logging to Langfuse: {config.host_url}", file=sys.stderr)


def setup_instrumentation_file(
    file: str,
) -> None:
    """Sets up OpenTelemetry instrumentation to log traces to a file."""
    provider = _create_tracer_provider()
    file_stream = open(file, "w", encoding="utf-8")
    exporter = ConsoleSpanExporter(out=file_stream)
    processor = SimpleSpanProcessor(exporter)
    provider.add_span_processor(processor)
    trace.set_tracer_provider(provider)
    atexit.register(file_stream.close)
    print(f"Telemetry logging to: {file}", file=sys.stderr)
