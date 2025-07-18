import os
import yaml
from typing import Literal, Optional, List, Dict
from enum import Enum
from pydantic import BaseModel, ValidationError, field_validator


class ProviderEnum(str, Enum):
    """Enumeration of supported LLM providers."""
    OLLAMA = "ollama"
    OPENAI = "openai"
    OPENROUTER = "openrouter"

class AgentConfig(BaseModel):
    instructions: str

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
    enabled: bool = True
    transport: Optional[Literal["sse", "http", "stdio"]] = "stdio"
    command: Optional[List[str]] = None # for stdio transport
    url: Optional[str] = None           # for sse and http transports
    tool_prefix: Optional[str] = None
    cwd: Optional[str] = None

class Config(BaseModel):
    agent: AgentConfig
    llm: LLMConfig
    mcp: Optional[Dict[str, MCPServerConfig]] = None

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







