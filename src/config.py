import os
import yaml
from typing import Any, Literal, Optional, List, Dict
from enum import Enum
from pydantic import BaseModel, ValidationError, field_validator


class ProviderEnum(str, Enum):
    """Enumeration of supported LLM providers."""
    OLLAMA = "ollama"
    OPENAI = "openai"
    OPENROUTER = "openrouter"
    LMSTUDIO = "lmstudio"

class AgentConfig(BaseModel):
    instructions: str
    output_type: Optional[Any] = str

class LLMConfig(BaseModel):
    model: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    timeout: Optional[float] = None

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
            if file_path.startswith("~"): file_path = os.path.expanduser(file_path)
            try:
                with open(file_path, 'r') as f:
                    return f.read().strip()
            except FileNotFoundError:
                raise ValueError(f"File '{file_path}' not found for '{v}'")
        return v

class MCPServerConfig(BaseModel):
    """Configuration for an MCP server tool/service."""
    enabled: bool = True
    transport: Optional[Literal["sse", "http", "stdio"]] = "stdio"
    command: Optional[List[str]] = None  # for stdio transport
    url: Optional[str] = None            # for sse and http transports
    tool_prefix: Optional[str] = None
    cwd: Optional[str] = None
    env: Optional[Dict[str, str]] = None

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

class Config(BaseModel):
    """Top-level configuration for the agent, LLM, and MCP tools/services."""
    agent: AgentConfig
    llm: LLMConfig
    mcp: Optional[dict[str, MCPServerConfig]] = None

def load_config(path: str) -> Config:
    try:
        with open(path, "r") as f:
            raw = yaml.safe_load(f)
    except FileNotFoundError:
        raise RuntimeError(f"Configuration file '{path}' not found.")
    except yaml.YAMLError as e:
        raise RuntimeError(f"YAML syntax error: {e}")
    except Exception as e:
        raise RuntimeError(f"Error loading config file: {e}")

    try:
        return Config(**raw)
    except ValidationError as e:
        raise RuntimeError(f"Config validation error: {e}")







