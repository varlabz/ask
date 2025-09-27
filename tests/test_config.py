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
