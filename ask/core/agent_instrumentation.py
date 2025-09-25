import io
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    SimpleSpanProcessor,
    ConsoleSpanExporter,
)
from opentelemetry.sdk.resources import Resource
import sys

def setup_instrumentation(service_name="pydantic-ai-agent", stream: io.TextIOWrapper | None = None):
    """
    Sets up OpenTelemetry instrumentation to log traces to the console or a stream.
    """
    resource = Resource(attributes={
        "service.name": service_name
    })

    provider = TracerProvider(resource=resource)

    # Add a ConsoleSpanExporter to print traces to the console or a file
    if stream:
        processor = SimpleSpanProcessor(ConsoleSpanExporter(out=stream))
        print(f"OpenTelemetry instrumentation set up to log to {stream.name}.", file=sys.stderr)
    else:
        processor = SimpleSpanProcessor(ConsoleSpanExporter())
        # print("OpenTelemetry instrumentation set up to log to console.", file=sys.stderr)

    provider.add_span_processor(processor)

    # Set the global TracerProvider
    trace.set_tracer_provider(provider)
