# ASK Agent - Python Package

A PydanticAI-based agent with Model Context Protocol (MCP) server support.

## Features

- **Multi-Provider LLM Support**: OpenAI, Ollama, OpenRouter
- **MCP Server Integration**: Support for stdio, SSE, and HTTP transports
- **Configuration Management**: YAML-based configuration with environment variable support
- **CLI Interface**: Command-line tools for running agents and MCP clients
- **Type Safety**: Built with Pydantic for robust type checking

## Installation

```bash
pip install -e .
```

## Quick Start

1. **Create a configuration file** (`.ask.yaml`):
```yaml
agent:
  instructions: "You are a helpful AI assistant."

llm:
  model: "ollama:qwen2.5:14b"
  base_url: "http://localhost:11434/v1"
  temperature: 0.1

mcp:
  fetch:
    enabled: true
    transport: "stdio"
    command: ["uvx", "mcp-server-fetch", "--ignore-robots-txt"]
```

2. **Run the agent**:
```bash
cli -c .ask.yaml "What is the weather like today?"
```

3. **Run MCP client**:
```bash
mcp -c .ask.yaml
```

## Configuration

### Agent Configuration
- `instructions`: System prompt for the agent

### LLM Configuration
- `model`: Format: `provider:model_name` (e.g., `"openai:gpt-4o"`, `"ollama:llama2"`)
- `api_key`: API key (supports `env/VARIABLE_NAME` format)
- `base_url`: Custom endpoint URL
- `temperature`: Sampling temperature (0.0 - 2.0)

### MCP Server Configuration
Each MCP server can be configured with:
- `enabled`: Whether to enable the server
- `transport`: Transport type (`stdio`, `sse`, `http`)
- `command`: Command array for stdio transport
- `url`: URL for SSE/HTTP transports
- `tool_prefix`: Prefix for tool names
- `cwd`: Working directory for stdio commands

## Project Structure

```
src/
├── __init__.py          # Package initialization
├── agent.py             # PydanticAI agent wrapper
├── cli.py               # CLI entry point
├── config.py            # Configuration models
├── mcp_cli.py           # MCP CLI entry point  
├── mcp_client.py        # MCP server management
└── model.py             # LLM model creation

tests/
├── test_config.py       # Configuration tests
├── test_llm_config.py   # LLM configuration tests
└── test_mcp_client.py   # MCP client tests
```

## Development

### Running Tests
```bash
python -m pytest tests/ -v
```

### Code Quality
The package includes:
- Type hints throughout
- Comprehensive test coverage
- Pydantic data validation
- Error handling with descriptive messages

## API Reference

### Core Functions

- `create_agent(config: Config) -> AgentASK`: Create a configured agent
- `run_agent(prompt: str, agent: AgentASK) -> str`: Run agent with prompt
- `load_config(path: str) -> Config`: Load configuration from YAML
- `create_model(llm_config: LLMConfig) -> OpenAIModel`: Create LLM model
- `create_mcp_servers(mcp_config: Dict[str, MCPServerConfig]) -> List[...]`: Create MCP servers

## License

This project is released under the MIT License.
