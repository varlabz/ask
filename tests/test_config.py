import os

import pytest

from ask.core.config import Config, load_config


def test_load_config_success(tmp_path, monkeypatch):
    # Prepare a config file with env var for api_key
    config_yaml = """
agent:
  instructions: "Test instructions."
llm:
  model: "openai:gpt-4o"
  api_key: "env:TEST_API_KEY"
"""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(config_yaml)
    monkeypatch.setenv("TEST_API_KEY", "dummy-key")
    cfg = load_config([str(config_path)])
    assert isinstance(cfg, Config)
    assert cfg.llm.api_key == "dummy-key"
    assert cfg.agent.instructions == "Test instructions."
    assert cfg.llm.model == "openai:gpt-4o"


def test_load_config_missing_env(monkeypatch, tmp_path):
    config_yaml = """
agent:
  instructions: "Test instructions."
llm:
  model: "openai:gpt-4o"
  api_key: "env:NOT_SET_ENV"
"""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(config_yaml)
    monkeypatch.delenv("NOT_SET_ENV", raising=False)
    with pytest.raises(RuntimeError) as exc:
        load_config([str(config_path)])
    assert "Environment variable 'NOT_SET_ENV' not set" in str(exc.value)


def test_load_config_file_not_found():
    with pytest.raises(RuntimeError) as exc:
        load_config(["/nonexistent/config.yaml"])
    assert "not found" in str(exc.value)


def test_load_config_yaml_error(tmp_path):
    config_yaml = "agent: [bad: yaml"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(config_yaml)
    with pytest.raises(RuntimeError) as exc:
        load_config([str(config_path)])
    assert "YAML syntax error" in str(exc.value)


def test_load_config_invalid_types(tmp_path, monkeypatch):
    config_yaml = """
agent:
  instructions: 12345  # Should be str
llm:
  model: 67890         # Should be str
  api_key: "env:TEST_API_KEY"
"""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(config_yaml)
    monkeypatch.setenv("TEST_API_KEY", "dummy-key")
    with pytest.raises(RuntimeError) as exc:
        load_config([str(config_path)])
    assert "Config validation error" in str(exc.value)


def test_load_config_missing_required(tmp_path):
    config_yaml = """
llm:
  model: "openai:gpt-4o"
  api_key: "env:TEST_API_KEY"
"""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(config_yaml)
    with pytest.raises(RuntimeError) as exc:
        load_config([str(config_path)])
    assert "Config validation error" in str(exc.value)


def test_load_config_extra_fields(tmp_path, monkeypatch):
    config_yaml = """
agent:
  instructions: "Test instructions."
  extra_field: "should not be here"
llm:
  model: "openai:gpt-4o"
  api_key: "env:TEST_API_KEY"
  another_extra: "nope"
"""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(config_yaml)
    monkeypatch.setenv("TEST_API_KEY", "dummy-key")
    # With extra fields forbidden, loading should raise a validation error
    with pytest.raises(RuntimeError) as exc:
        load_config([str(config_path)])
    assert "Config validation error" in str(exc.value)


def test_mcp_env_field_success(tmp_path, monkeypatch):
    """Test MCPServerConfig env field loads and validates correctly."""
    config_yaml = """
agent:
  instructions: "Test instructions."
llm:
  model: "openai:gpt-4o"
  api_key: "env:TEST_API_KEY"
mcp:
  fetch:
    enabled: true
    command: ["uvx", "mcp-server-fetch"]
    env:
      FOO: "bar"
      BAZ: "qux"
"""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(config_yaml)
    monkeypatch.setenv("TEST_API_KEY", "dummy-key")
    cfg = load_config([str(config_path)])
    assert isinstance(cfg, Config)
    assert cfg.mcp is not None
    assert cfg.mcp["fetch"].env == {"FOO": "bar", "BAZ": "qux"}


def test_mcp_env_field_invalid_type(tmp_path, monkeypatch):
    """Test MCPServerConfig env field with invalid type (not a dict)."""
    config_yaml = """
agent:
  instructions: "Test instructions."
llm:
  model: "openai:gpt-4o"
  api_key: "env:TEST_API_KEY"
mcp:
  fetch:
    enabled: true
    command: ["uvx", "mcp-server-fetch"]
    env: "not-a-dict"
"""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(config_yaml)
    monkeypatch.setenv("TEST_API_KEY", "dummy-key")
    with pytest.raises(RuntimeError) as exc:
        load_config([str(config_path)])
    assert "env must be a dictionary" in str(exc.value)


def test_mcp_env_field_non_str_key_value(tmp_path, monkeypatch):
    """Test MCPServerConfig env field with non-string key or value."""
    config_yaml = """
agent:
  instructions: "Test instructions."
llm:
  model: "openai:gpt-4o"
  api_key: "env:TEST_API_KEY"
mcp:
  fetch:
    enabled: true
    command: ["uvx", "mcp-server-fetch"]
    env:
      123: "bar"
      FOO: 456
"""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(config_yaml)
    monkeypatch.setenv("TEST_API_KEY", "dummy-key")
    with pytest.raises(RuntimeError) as exc:
        load_config([str(config_path)])
    assert "env keys and values must be strings" in str(exc.value)


def test_load_config_merge_multiple_files(tmp_path, monkeypatch):
    """Test that multiple config files are merged, with later files overriding earlier ones."""
    # First config file
    config1_yaml = """
agent:
  instructions: "First instructions."
  name: "First Agent"
llm:
  model: "openai:gpt-3.5-turbo"
  api_key: "env:TEST_API_KEY"
  temperature: 0.5
"""
    config1_path = tmp_path / "config1.yaml"
    config1_path.write_text(config1_yaml)

    # Second config file (overrides some top-level keys)
    config2_yaml = """
agent:
  instructions: "Second instructions."
  name: "Second Agent"
"""
    config2_path = tmp_path / "config2.yaml"
    config2_path.write_text(config2_yaml)

    monkeypatch.setenv("TEST_API_KEY", "dummy-key")
    cfg = load_config([str(config1_path), str(config2_path)])

    assert isinstance(cfg, Config)
    # Should use overridden values from config2 (top-level merge means entire 'agent' is replaced)
    assert cfg.agent.instructions == "Second instructions."
    assert cfg.agent.name == "Second Agent"
    # LLM section should be preserved from config1 since config2 doesn't have it
    assert cfg.llm.model == "openai:gpt-3.5-turbo"
    assert cfg.llm.temperature == 0.5
    assert cfg.llm.api_key == "dummy-key"


def test_load_config_skip_none_paths(tmp_path, monkeypatch):
    """Test that None paths are skipped in the list."""
    config_yaml = """
agent:
  instructions: "Test instructions."
llm:
  model: "openai:gpt-4o"
  api_key: "env:TEST_API_KEY"
"""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(config_yaml)
    monkeypatch.setenv("TEST_API_KEY", "dummy-key")

    # Should not raise an error even with None in the list
    cfg = load_config([None, str(config_path), None])
    assert isinstance(cfg, Config)
    assert cfg.llm.api_key == "dummy-key"


def test_load_config_with_specific_type_llm(tmp_path, monkeypatch):
    """Test loading a specific config type (LLMConfig) instead of full Config."""
    from ask.core.config import LLMConfig

    config_yaml = """
agent:
  instructions: "Test instructions."
llm:
  model: "openai:gpt-4o"
  api_key: "env:TEST_API_KEY"
  temperature: 0.7
  max_tokens: 2000
"""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(config_yaml)
    monkeypatch.setenv("TEST_API_KEY", "dummy-key")

    llm_cfg = load_config([str(config_path)], type=LLMConfig)
    assert isinstance(llm_cfg, LLMConfig)
    assert llm_cfg.model == "openai:gpt-4o"
    assert llm_cfg.api_key == "dummy-key"
    assert llm_cfg.temperature == 0.7
    assert llm_cfg.max_tokens == 2000


def test_load_config_with_specific_type_agent(tmp_path, monkeypatch):
    """Test loading a specific config type (AgentConfig) instead of full Config."""
    from ask.core.config import AgentConfig

    config_yaml = """
agent:
  name: "Test Agent"
  instructions: "Test instructions."
llm:
  model: "openai:gpt-4o"
  api_key: "env:TEST_API_KEY"
"""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(config_yaml)
    monkeypatch.setenv("TEST_API_KEY", "dummy-key")

    agent_cfg = load_config([str(config_path)], type=AgentConfig)
    assert isinstance(agent_cfg, AgentConfig)
    assert agent_cfg.name == "Test Agent"
    assert agent_cfg.instructions == "Test instructions."


def test_load_config_with_specific_type_missing_key(tmp_path):
    """Test that loading a specific type fails if the required key is missing."""
    from ask.core.config import LLMConfig

    config_yaml = """
agent:
  instructions: "Test instructions."
"""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(config_yaml)

    with pytest.raises(RuntimeError) as exc:
        load_config([str(config_path)], type=LLMConfig)
    assert "Key 'llm' not found in config" in str(exc.value)


def test_load_config_with_unsupported_type(tmp_path, monkeypatch):
    """Test that loading with an unsupported type raises an error."""
    config_yaml = """
agent:
  instructions: "Test instructions."
llm:
  model: "openai:gpt-4o"
  api_key: "env:TEST_API_KEY"
"""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(config_yaml)
    monkeypatch.setenv("TEST_API_KEY", "dummy-key")

    # Create a dummy type that's not in Config
    class UnsupportedType:
        pass

    with pytest.raises(RuntimeError) as exc:
        load_config([str(config_path)], type=UnsupportedType)
    assert "Type" in str(exc.value) and "not found in Config" in str(exc.value)


def test_load_config_non_dict_root(tmp_path):
    """Test that a config file with non-dict root raises an error."""
    config_yaml = """
- item1
- item2
- item3
"""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(config_yaml)

    with pytest.raises(ValueError) as exc:
        load_config([str(config_path)])
    assert "must contain a dictionary at the root" in str(exc.value)


def test_load_config_api_key_from_file(tmp_path):
    """Test loading api_key from a file."""
    # Create a file with the API key
    key_file = tmp_path / "api_key.txt"
    key_file.write_text("secret-api-key-from-file")

    config_yaml = f"""
agent:
  instructions: "Test instructions."
llm:
  model: "openai:gpt-4o"
  api_key: "file:{key_file}"
"""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(config_yaml)

    cfg = load_config([str(config_path)])
    assert isinstance(cfg, Config)
    assert cfg.llm.api_key == "secret-api-key-from-file"


def test_load_config_api_key_file_not_found(tmp_path):
    """Test that loading api_key from non-existent file raises an error."""
    config_yaml = """
agent:
  instructions: "Test instructions."
llm:
  model: "openai:gpt-4o"
  api_key: "file:/nonexistent/key.txt"
"""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(config_yaml)

    with pytest.raises(RuntimeError) as exc:
        load_config([str(config_path)])
    assert "File '/nonexistent/key.txt' not found" in str(exc.value)


def test_load_config_api_key_file_with_tilde(tmp_path, monkeypatch):
    """Test loading api_key from a file with tilde expansion."""
    # Create a temp file that we'll reference
    key_file = tmp_path / "api_key.txt"
    key_file.write_text("secret-key-with-tilde")

    # Mock expanduser to return our temp file when called with the api_key path
    original_expanduser = os.path.expanduser

    def mock_expanduser(path):
        if path == "~/api_key.txt":
            return str(key_file)
        return original_expanduser(path)

    monkeypatch.setattr("os.path.expanduser", mock_expanduser)

    config_yaml = """
agent:
  instructions: "Test instructions."
llm:
  model: "openai:gpt-4o"
  api_key: "file:~/api_key.txt"
"""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(config_yaml)

    cfg = load_config([str(config_path)])
    assert isinstance(cfg, Config)
    assert cfg.llm.api_key == "secret-key-with-tilde"


def test_load_config_merge_preserves_first_file_values(tmp_path, monkeypatch):
    """Test that merging preserves values from first file when not overridden."""
    config1_yaml = """
agent:
  name: "Agent One"
  instructions: "First instructions."
llm:
  model: "openai:gpt-3.5-turbo"
  api_key: "env:TEST_API_KEY"
  temperature: 0.5
  max_tokens: 500
mcp:
  server1:
    enabled: true
    command: ["cmd1"]
"""
    config1_path = tmp_path / "config1.yaml"
    config1_path.write_text(config1_yaml)

    config2_yaml = """
mcp:
  server2:
    enabled: false
    command: ["cmd2"]
"""
    config2_path = tmp_path / "config2.yaml"
    config2_path.write_text(config2_yaml)

    monkeypatch.setenv("TEST_API_KEY", "dummy-key")
    cfg = load_config([str(config1_path), str(config2_path)])

    # Values from first file should be preserved when not in second file
    assert cfg.agent.name == "Agent One"
    assert cfg.agent.instructions == "First instructions."
    assert cfg.llm.model == "openai:gpt-3.5-turbo"
    assert cfg.llm.temperature == 0.5
    assert cfg.llm.max_tokens == 500

    # MCP servers should be completely replaced (top-level merge)
    assert cfg.mcp is not None
    assert "server1" not in cfg.mcp
    assert "server2" in cfg.mcp
    assert cfg.mcp["server2"].enabled is False


def test_load_config_with_specific_type_embedder(tmp_path, monkeypatch):
    """Test loading EmbedderConfig type."""
    from ask.core.config import EmbedderConfig

    config_yaml = """
agent:
  instructions: "Test instructions."
llm:
  model: "openai:gpt-4o"
  api_key: "env:TEST_API_KEY"
embedder:
  model: "openai:text-embedding-3-small"
  api_key: "env:EMBED_API_KEY"
  base_url: "https://api.openai.com/v1"
"""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(config_yaml)
    monkeypatch.setenv("TEST_API_KEY", "dummy-key")
    monkeypatch.setenv("EMBED_API_KEY", "embed-key")

    embedder_cfg = load_config([str(config_path)], type=EmbedderConfig)
    assert isinstance(embedder_cfg, EmbedderConfig)
    assert embedder_cfg.model == "openai:text-embedding-3-small"
    assert embedder_cfg.api_key == "embed-key"
    assert embedder_cfg.base_url == "https://api.openai.com/v1"


def test_load_config_with_specific_type_server(tmp_path, monkeypatch):
    """Test loading ServerConfig type."""
    from ask.core.config import ServerConfig

    config_yaml = """
agent:
  instructions: "Test instructions."
llm:
  model: "openai:gpt-4o"
  api_key: "env:TEST_API_KEY"
server:
  name: "My ASK Server"
  transport: "http"
  debug: true
  port: 9000
  log_level: "DEBUG"
  tool_name: "my_ask_tool"
"""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(config_yaml)
    monkeypatch.setenv("TEST_API_KEY", "dummy-key")

    server_cfg = load_config([str(config_path)], type=ServerConfig)
    assert isinstance(server_cfg, ServerConfig)
    assert server_cfg.name == "My ASK Server"
    assert server_cfg.transport == "http"
    assert server_cfg.debug is True
    assert server_cfg.port == 9000
    assert server_cfg.log_level == "DEBUG"
    assert server_cfg.tool_name == "my_ask_tool"


def test_load_config_with_specific_type_trace(tmp_path, monkeypatch):
    """Test loading TraceConfig type."""
    from ask.core.config import TraceConfig

    config_yaml = """
agent:
  instructions: "Test instructions."
llm:
  model: "openai:gpt-4o"
  api_key: "env:TEST_API_KEY"
trace:
  public_key: "env:TRACE_PUBLIC_KEY"
  secret_key: "env:TRACE_SECRET_KEY"
  host_url: "https://cloud.langfuse.com"
"""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(config_yaml)
    monkeypatch.setenv("TEST_API_KEY", "dummy-key")
    monkeypatch.setenv("TRACE_PUBLIC_KEY", "pk-test-123")
    monkeypatch.setenv("TRACE_SECRET_KEY", "sk-test-456")

    trace_cfg = load_config([str(config_path)], type=TraceConfig)
    assert isinstance(trace_cfg, TraceConfig)
    assert trace_cfg.public_key == "pk-test-123"
    assert trace_cfg.secret_key == "sk-test-456"
    assert trace_cfg.host_url == "https://cloud.langfuse.com"


def test_load_config_with_specific_type_merge_multiple_files(tmp_path, monkeypatch):
    """Test loading a specific type with multiple files merged."""
    from ask.core.config import LLMConfig

    config1_yaml = """
agent:
  instructions: "Test instructions."
llm:
  model: "openai:gpt-3.5-turbo"
  api_key: "env:TEST_API_KEY"
  temperature: 0.5
"""
    config1_path = tmp_path / "config1.yaml"
    config1_path.write_text(config1_yaml)

    config2_yaml = """
llm:
  model: "openai:gpt-4o"
  max_tokens: 2000
  timeout: 30.0
"""
    config2_path = tmp_path / "config2.yaml"
    config2_path.write_text(config2_yaml)

    monkeypatch.setenv("TEST_API_KEY", "dummy-key")
    llm_cfg = load_config([str(config1_path), str(config2_path)], type=LLMConfig)

    assert isinstance(llm_cfg, LLMConfig)
    # Second file should completely override the llm section
    assert llm_cfg.model == "openai:gpt-4o"
    assert llm_cfg.max_tokens == 2000
    assert llm_cfg.timeout == 30.0
    # These were only in first file's llm section, so won't be present after override
    assert llm_cfg.api_key is None  # Default value
    assert llm_cfg.temperature is None  # Default value
