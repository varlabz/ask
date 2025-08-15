
# ASK Agent Swiss Knife

**PydanticAI-powered CLI agent with Model Context Protocol (MCP) server support**

ASK is a versatile AI agent that works both as a CLI tool with MCP server integrations and as an MCP server itself to extend other LLMs like Claude and VS Code Copilot.

## Features

- **Multi-provider LLM support**: OpenAI, Ollama, OpenRouter, LMStudio, Google, Anthropic or any OpenAI compatible model API
- **MCP server integration**: stdio, SSE, HTTP transports
- **Dual mode operation**: CLI agent + MCP server
- **Rich tool ecosystem**: Web search, file ops, memory, YouTube transcripts, any MCP tool
- **Environment variable support**: Secure API key management
- **YAML configuration**: Simple, readable config format

## Quick Usage with uvx

Use ASK directly without installation:

```bash
uvx --from git+https://github.com/varlabz/ask cli "What is Python?"
```

With with a simple config:

```bash
# Create minimal config
echo "agent:
  instructions: 'You are a helpful AI assistant.'
llm:
  model: 'ollama:qwen2.5:14b'
  base_url: 'http://localhost:11434/v1'
  temperature: 0.1" > .ask.yaml

# Run with uvx
uvx --from git+https://github.com/varlabz/ask cli -c .ask.yaml "Explain machine learning"
```

## Configuration

### Example of advanced configuration

Create a `agent.yaml` file:

```yaml
agent:
  instructions: "You are a helpful AI assistant with access to web search and file operations."

llm:
  model: "ollama:qwen2.5:14b"
  base_url: "http://localhost:11434/v1"
  temperature: 0.1

mcp:
  fetch:
    command: ["uvx", "mcp-server-fetch", "--ignore-robots-txt"]
  
  search:
    command: ["uvx", "--from", "git+https://github.com/varlabz/searxng-mcp", "searxng-mcp"]
```

### Complete Configuration Example

```yaml
agent:
  instructions: |
    You are a helpful AI assistant with access to various tools and services.
    Provide accurate, helpful, and concise responses. When using tools, explain
    what you're doing and why. Be proactive in suggesting useful tools when appropriate.

llm:
  model: "openai:gpt-4o"
  api_key: "env/OPENAI_API_KEY"
  # Alternative providers:
  # model: "openrouter:anthropic/claude-3.5-sonnet"
  # api_key: "env:OPENROUTER_API_KEY"
  # or
  # api_key: "file:path to openrouter key file"
  # or
  # api_key: "api key"

mcp:
  filesystem:
    command: ["npx", "-y", "@modelcontextprotocol/server-filesystem", "."]
  
  memory:
    command: ["npx", "-y", "@modelcontextprotocol/server-memory"]
  
  fetch:
    command: ["uvx", "mcp-server-fetch", "--ignore-robots-txt"]
  
  youtube:
    command: ["npx", "-y", "https://github.com/varlabz/youtube-mcp", "--mcp"]
    
  sequential_thinking:
    command: ["npx", "-y", "@modelcontextprotocol/server-sequential-thinking"]
  
  searxng:
    command: ["uvx", "--from", "git+https://github.com/varlabz/searxng-mcp", "searxng-mcp"]
```

## Use as MCP Server

ASK can extend other LLMs by running as an MCP server, providing access to your configured AI agent.

### Claude Desktop Configuration

Add to your Claude Desktop config (`~/Library/Application Support/Claude/claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "ask": {
      "command": "uvx",
      "args": [
        "--from",
        "git+https://github.com/varlabz/ask",
        "ask-mcp",
        "-c",
        "/path/to/your/.ask.yaml"
      ]
    }
  }
}
```

### VS Code MCP Extension Configuration

Add to VS Code settings (`mcp.json`):

```json
{
  "mcp_servers": {
    "ask": {
      "command": "uvx",
      "args": [
        "--from", 
        "git+https://github.com/varlabz/ask",
        "ask-mcp",
        "-c",
        "${workspaceFolder}/.ask.yaml"
      ]
    }
  }
}
```

## Development Setup

### Clone and Setup

```bash
git clone https://github.com/varlabz/ask.git
cd ask
uv sync --dev
```

### Run Tests

```bash
pytest 
```

### Local Installation

```bash
pip install -e .
```

## Environment Variables

Set API keys as environment variables (not recommended):

```bash
export OPENAI_API_KEY="your-openai-key"
export OPENROUTER_API_KEY="your-openrouter-key"
export SEARX_HOST="http://localhost:8080"
```

Reference them in config:

```yaml
llm:
  model: "openai:gpt-4o"
  api_key: "env/OPENAI_API_KEY"
```

