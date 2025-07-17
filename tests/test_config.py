import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import pytest
from pydantic import ValidationError
from config import load_config, Config

def test_load_config_success(tmp_path, monkeypatch):
    # Prepare a config file with env var for api_key
    config_yaml = '''
agent:
  instructions: "Test instructions."
llm:
  model: "openai:gpt-4o"
  api_key: "env:TEST_API_KEY"
'''
    config_path = tmp_path / "config.yaml"
    config_path.write_text(config_yaml)
    monkeypatch.setenv("TEST_API_KEY", "dummy-key")
    cfg = load_config(str(config_path))
    assert isinstance(cfg, Config)
    assert cfg.llm.api_key == "dummy-key"
    assert cfg.agent.instructions == "Test instructions."
    assert cfg.llm.model == "openai:gpt-4o"

def test_load_config_missing_env(monkeypatch, tmp_path):
    config_yaml = '''
agent:
  instructions: "Test instructions."
llm:
  model: "openai:gpt-4o"
  api_key: "env:NOT_SET_ENV"
'''
    config_path = tmp_path / "config.yaml"
    config_path.write_text(config_yaml)
    monkeypatch.delenv("NOT_SET_ENV", raising=False)
    with pytest.raises(RuntimeError) as exc:
        load_config(str(config_path))
    assert "Environment variable 'NOT_SET_ENV' not set" in str(exc.value)

def test_load_config_file_not_found():
    with pytest.raises(RuntimeError) as exc:
        load_config("/nonexistent/config.yaml")
    assert "not found" in str(exc.value)

def test_load_config_yaml_error(tmp_path):
    config_yaml = 'agent: [bad: yaml'
    config_path = tmp_path / "config.yaml"
    config_path.write_text(config_yaml)
    with pytest.raises(RuntimeError) as exc:
        load_config(str(config_path))
    assert "YAML syntax error" in str(exc.value)

def test_load_config_invalid_types(tmp_path, monkeypatch):
    config_yaml = '''
agent:
  instructions: 12345  # Should be str
llm:
  model: 67890         # Should be str
  api_key: "env:TEST_API_KEY"
'''
    config_path = tmp_path / "config.yaml"
    config_path.write_text(config_yaml)
    monkeypatch.setenv("TEST_API_KEY", "dummy-key")
    with pytest.raises(RuntimeError) as exc:
        load_config(str(config_path))
    assert "Config validation error" in str(exc.value)


def test_load_config_missing_required(tmp_path):
    config_yaml = '''
llm:
  model: "openai:gpt-4o"
  api_key: "env:TEST_API_KEY"
'''
    config_path = tmp_path / "config.yaml"
    config_path.write_text(config_yaml)
    with pytest.raises(RuntimeError) as exc:
        load_config(str(config_path))
    assert "Config validation error" in str(exc.value)


def test_load_config_extra_fields(tmp_path, monkeypatch):
    config_yaml = '''
agent:
  instructions: "Test instructions."
  extra_field: "should not be here"
llm:
  model: "openai:gpt-4o"
  api_key: "env:TEST_API_KEY"
  another_extra: "nope"
'''
    config_path = tmp_path / "config.yaml"
    config_path.write_text(config_yaml)
    monkeypatch.setenv("TEST_API_KEY", "dummy-key")
    # Pydantic by default ignores extra fields unless you set extra=forbid
    # To make this a true negative test, update the Config model to use extra=forbid
    # For now, this test will pass if config loads and extra fields are ignored
    cfg = load_config(str(config_path))
    assert isinstance(cfg, Config)
    assert cfg.agent.instructions == "Test instructions."
    assert cfg.llm.api_key == "dummy-key"
    assert cfg.llm.model == "openai:gpt-4o"
