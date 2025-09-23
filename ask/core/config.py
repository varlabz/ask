import os
import yaml
import builtins
import typing
from typing import Any, Literal, Optional, List, Dict
from enum import Enum
from pydantic import BaseModel, ValidationError, field_validator, ConfigDict, field_serializer

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
    output_type: Any = str
    # Forbid unknown fields
    model_config = ConfigDict(extra="forbid")

    @field_validator("output_type", mode="before")
    @classmethod
    def convert_output_type(cls, v: Any) -> type:
        """Convert string type descriptions to Python types."""
        if v is None:
            return str
        if isinstance(v, type) or hasattr(v, '__origin__'):
            return v
        if not isinstance(v, str):
            return v

        safe_namespace = {
            **{name: getattr(typing, name) for name in typing.__all__},
            **{name: getattr(builtins, name) for name in dir(builtins)}
        }
        try:
            return eval(v, {"__builtins__": {}}, safe_namespace)
        except (NameError, SyntaxError) as e:
            raise ValueError(f"Unknown or invalid type string: {v}") from e

    @field_serializer('output_type')
    def serialize_output_type(self, value: type) -> str:
        """Serialize the output_type to a string for JSON compatibility."""
        if hasattr(value, '__name__'):
            return value.__name__
        return str(value)

class LLMConfig(BaseModel):
    model: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    timeout: Optional[float] = None
    max_history: int = 0        # 0 - no history, >0 - keep summary in ~N of words. more means more context
    compress_history: bool = True  # whether to clean up history messages to save tokens
    # Forbid unknown fields
    model_config = ConfigDict(extra="forbid")

    @field_validator("api_key", mode="before")
    def resolve_api_key(cls, v):
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
                with open(file_path, 'r') as f:
                    return f.read().strip()
            except FileNotFoundError:
                raise ValueError(f"File '{file_path}' not found for '{v}'")
        return v

class MCPServerConfig(BaseModel):
    """Configuration for an MCP server tool/service."""
    enabled: bool = True
    transport: Literal["stdio", "sse", "streamable-http", "http"] = "stdio"
    command: Optional[List[str]] = None  # for stdio transport
    url: Optional[str] = None            # for sse and http transports
    tool_prefix: Optional[str] = None
    cwd: Optional[str] = None
    env: Optional[Dict[str, str]] = None
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
    instructions: Optional[str] = None
    transport: Literal["stdio", "sse", "streamable-http"] = "stdio"
    debug: bool = False
    port: int = 8000
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "ERROR"
    # Forbid unknown fields
    model_config = ConfigDict(extra="forbid")

class Config(BaseModel):
    """Top-level configuration for the agent, LLM, and MCP tools/services."""
    agent: AgentConfig
    llm: LLMConfig
    mcp: Optional[dict[str, MCPServerConfig]] = None
    server: Optional[ServerConfig] = None
    # Forbid unknown fields
    model_config = ConfigDict(extra="forbid")

def load_config(paths: List[str]) -> Config:
    """Deep-merge multiple YAML config files into a `Config`. Later files override."""
    merged_raw: dict = {}
    for p in paths:
        if p is None:   # skip empty paths
            continue

        try:
            with open(os.path.expanduser(p), "r") as f:
                raw = yaml.safe_load(f)
                if not isinstance(raw, dict):
                    raise ValueError(f"Config file '{p}' must contain a dictionary at the root.")
                # merge top level keys only
                merged_raw = {**merged_raw, **raw}
        except FileNotFoundError:
            raise RuntimeError(f"Configuration file '{p}' not found.")
        except yaml.YAMLError as e:
            raise RuntimeError(f"YAML syntax error in '{p}': {e}")
        except ValueError:
            raise
        except Exception as e:
            raise RuntimeError(f"Error loading config file '{p}': {e}")
    try:
        return Config(**merged_raw)
    except ValidationError as e:
        raise RuntimeError(f"Config validation error: {e}")

def load_config_dict(config_dict: dict) -> Config:
    """Load configuration from a dictionary.

    Args:
        config_dict: Dictionary containing configuration data.

    Returns:
        Config: Merged configuration object.

    Raises:
        RuntimeError: If the config is invalid.
    """
    try:
        return Config(**config_dict)
    except ValidationError as e:
        raise RuntimeError(f"Config validation error: {e}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python config.py <config_file1> [<config_file2> ...]")
        sys.exit(1)

    config = load_config(sys.argv[1:])
    print(config.model_dump_json(indent=2))
