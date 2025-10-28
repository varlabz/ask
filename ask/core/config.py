import builtins
import os
import typing
from enum import Enum
from typing import Any, Literal

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    ValidationError,
    field_serializer,
    field_validator,
)


class ProviderEnum(str, Enum):
    """Enumeration of supported LLM providers."""

    OLLAMA = "ollama"
    OPENAI = "openai"
    OPENROUTER = "openrouter"
    LMSTUDIO = "lmstudio"
    GOOGLE = "google"
    ANTHROPIC = "anthropic"


class AgentConfig(BaseModel):
    name: str = "ASK Agent"
    instructions: str
    output_type: Any = (
        str  # can be a string like "str", "int", "List[str]", or a Python type
    )
    input_type: Any = (
        str  # can be a string like "str", "int", "List[str]", or a Python type
    )
    # Forbid unknown fields
    model_config = ConfigDict(extra="forbid")

    @field_validator("output_type", mode="before")
    @classmethod
    def convert_output_type(cls, v: Any) -> type:
        """Convert string type descriptions to Python types."""
        if v is None:
            return str
        if isinstance(v, type) or hasattr(v, "__origin__"):
            return v
        if not isinstance(v, str):
            return v

        safe_namespace = {
            **{name: getattr(typing, name) for name in typing.__all__},
            **{name: getattr(builtins, name) for name in dir(builtins)},
        }
        try:
            return eval(v, {"__builtins__": {}}, safe_namespace)
        except (NameError, SyntaxError) as e:
            raise ValueError(f"Unknown or invalid type string: {v}") from e

    @field_serializer("output_type")
    def serialize_output_type(self, value: type) -> str:
        """Serialize the output_type to a string for JSON compatibility."""
        if hasattr(value, "__name__"):
            return value.__name__
        return str(value)


class LLMConfig(BaseModel):
    model: str  # e.g., "openai:gpt-4", "google:gemini-pro"
    api_key: str | None = (
        None  # can be "env:VAR_NAME" or "file:/path/to/file" or actual key
    )
    base_url: str | None = None  # for custom endpoints, e.g. local LLM server
    temperature: float | None = None  # 0.0 to 1.0
    max_tokens: int | None = None  # max tokens for response
    timeout: float | None = None  # in seconds
    max_history: int = (
        0  # 0 - no history, >0 - keep summary in ~N of words. more means more context
    )
    compress_history: bool = True  # whether to clean up history messages to save tokens
    use_tools: bool = True  # whether to enable tool use by LLM
    # Forbid unknown fields
    model_config = ConfigDict(extra="forbid")

    @field_validator("api_key", mode="before")
    def resolve_api_key(cls, v):
        return _resolve_api_key(v)


class EmbedderConfig(BaseModel):
    model: str  # e.g., "openai:gpt-4", "google:gemini-pro"
    api_key: str | None = (
        None  # can be "env:VAR_NAME" or "file:/path/to/file" or actual key
    )
    base_url: str | None = None  # for custom endpoints, e.g. local LLM server
    # Forbid unknown fields
    model_config = ConfigDict(extra="forbid")

    @field_validator("api_key", mode="before")
    def resolve_api_key(cls, v):
        return _resolve_api_key(v)


class MCPServerConfig(BaseModel):
    """Configuration for an MCP server tool/service."""

    enabled: bool = True
    transport: Literal["stdio", "sse", "http"] = "stdio"
    command: list[str] | None = None  # for stdio transport
    url: str | None = None  # for sse and http transports
    tool_prefix: str | None = None
    cwd: str | None = None
    env: dict[str, str] | None = None
    # Forbid unknown fields
    model_config = ConfigDict(extra="forbid")

    @field_validator("env", mode="before")
    @classmethod
    def validate_env(cls, v):
        """Validate that env is a dict of str to str, or None."""
        if v is None:
            return None
        if not isinstance(v, dict):
            raise ValueError("env must be a dictionary of str to str")
        for k, val in v.items():
            if not isinstance(k, str) or not isinstance(val, str):
                raise ValueError("env keys and values must be strings")
        return v

class ServerConfig(BaseModel):  # for running ask as server
    name: str = "ASK Server"
    instructions: str | None = None
    transport: Literal["stdio", "sse", "http"] = "stdio"
    debug: bool = False
    port: int = 8000
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "ERROR"
    tool_name: str = "ask"
    # Forbid unknown fields
    model_config = ConfigDict(extra="forbid")


class TraceConfig(BaseModel):  # configuration for Langfuse tracing
    public_key: str  # can be "env:VAR_NAME" or "file:/path/to/file" or actual key
    secret_key: str  # can be "env:VAR_NAME" or "file:/path/to/file" or actual key
    host_url: str = "https://localhost:3000"
    # Forbid unknown fields
    model_config = ConfigDict(extra="forbid")

    @field_validator("public_key", mode="before")
    @classmethod
    def resolve_api_key(cls, v):
        return _resolve_api_key(v)

    @field_validator("secret_key", mode="before")
    @classmethod
    def resolve_secret_api_key(cls, v):
        return _resolve_api_key(v)

ToolConfig = dict[str, Any] | None

class Config(BaseModel):
    """Top-level configuration for the agent, LLM, and MCP tools/services."""

    agent: AgentConfig
    llm: LLMConfig
    embedder: EmbedderConfig | None = None
    tools: dict[str, ToolConfig] | None = None  # list of predefined tools
    mcp: dict[str, MCPServerConfig] | None = None
    server: ServerConfig | None = None
    trace: TraceConfig | None = None
    # Forbid unknown fields
    model_config = ConfigDict(extra="forbid")


def _resolve_api_key(v):
    if isinstance(v, str) and v.startswith("env:"):
        env_var = v[4:]
        env_val = os.environ.get(env_var)
        if env_val is None:
            raise ValueError(f"Environment variable '{env_var}' not set for '{v}'")
        return env_val
    elif isinstance(v, str) and v.startswith("file:"):
        file_path = v[5:]
        # if file start with a tilde, expand it
        file_path = os.path.expanduser(file_path)
        try:
            with open(file_path) as f:
                return f.read().strip()
        except FileNotFoundError as e:
            raise ValueError(f"File '{file_path}' not found for '{v}'") from e
    return v


def load_config[Type](
    paths: list[str], type: type[Type] = Config, key: str | None = None
) -> Type:
    """Merge multiple YAML config files into a `Config`. Later files override."""
    merged_raw: dict = {}
    for p in paths:
        if p is None:  # skip empty paths
            continue

        try:
            with open(os.path.expanduser(p)) as f:
                raw = yaml.safe_load(f)
                if not isinstance(raw, dict):
                    raise ValueError(
                        f"Config file '{p}' must contain a dictionary at the root."
                    )
                # merge top level keys only
                merged_raw = {**merged_raw, **raw}
        except FileNotFoundError as e:
            raise RuntimeError(f"Configuration file '{p}' not found.") from e
        except yaml.YAMLError as e:
            raise RuntimeError(f"YAML syntax error in '{p}': {e}") from e
        except ValueError:
            raise
        except Exception as e:
            raise RuntimeError(f"Error loading config file '{p}': {e}") from e
    try:
        if key is not None:
            if key not in merged_raw:
                raise RuntimeError(f"Key '{key}' not found in merged config.")
            return type(**merged_raw[key])

        return type(**merged_raw)
    except ValidationError as e:
        raise RuntimeError(f"Config validation error: {e}") from e


def load_config_dict[Type](config_dict: dict, model_type: type[Type] = Config) -> Type:
    """Load configuration data into a specific Pydantic model."""
    try:
        return model_type(**config_dict)
    except ValidationError as e:
        raise RuntimeError(f"Config validation error: {e}") from e


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python config.py <config_file1> [<config_file2> ...]")
        sys.exit(1)

    config = load_config(sys.argv[1:])
    print(config.model_dump_json(indent=2))
