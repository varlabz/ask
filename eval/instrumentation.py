import os
from pathlib import Path

from langfuse import Langfuse, get_client
from pydantic_ai import Agent

from ask.core.config import load_config


def _here(filename: str) -> str:
    return str(Path(__file__).parent / filename)


def setup_instrumentation() -> Langfuse:
    """Sets up OpenTelemetry instrumentation with Langfuse."""
    config = load_config([_here(".trace.yaml")]).trace
    if config is None:
        raise RuntimeError("Trace configuration is missing in .trace.yaml")

    os.environ["LANGFUSE_PUBLIC_KEY"] = config.public_key
    os.environ["LANGFUSE_SECRET_KEY"] = config.secret_key
    os.environ["LANGFUSE_HOST"] = config.host_url
    ret = get_client()  # Initialize the Langfuse client
    Agent.instrument_all()
    return ret
