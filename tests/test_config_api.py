import os
import tempfile
import yaml
import pytest
from core.config import load_config, Config, AgentConfig, LLMConfig, MCPServerConfig

def write_yaml_file(data: dict) -> str:
    fd, path = tempfile.mkstemp(suffix=".yaml")
    with os.fdopen(fd, "w") as f:
        yaml.dump(data, f)
    return path

def test_load_config_single_file():
    config_data = {
        "agent": {"instructions": "Test agent", "output_type": "str"},
        "llm": {"model": "openai/gpt-4", "api_key": "test-key"},
        "mcp": None
    }
    path = write_yaml_file(config_data)
    cfg = load_config([path])
    assert isinstance(cfg, Config)
    assert cfg.agent.instructions == "Test agent"
    assert cfg.llm.model == "openai/gpt-4"
    os.remove(path)

def test_load_config_multiple_files():
    base_data = {
        "agent": {"instructions": "Base agent", "output_type": "str"},
        "llm": {"model": "openai/gpt-3", "api_key": "base-key"}
    }
    override_data = {
        "llm": {"model": "openai/gpt-4", "api_key": "override-key"}
    }
    path1 = write_yaml_file(base_data)
    path2 = write_yaml_file(override_data)
    cfg = load_config([path1, path2])
    assert cfg.llm.model == "openai/gpt-4"
    assert cfg.llm.api_key == "override-key"
    assert cfg.agent.instructions == "Base agent"
    os.remove(path1)
    os.remove(path2)

def test_load_config_file_not_found():
    with pytest.raises(RuntimeError) as exc:
        load_config(["/tmp/nonexistent.yaml"])
    assert "not found" in str(exc.value)

def test_load_config_invalid_yaml():
    fd, path = tempfile.mkstemp(suffix=".yaml")
    with os.fdopen(fd, "w") as f:
        f.write(": invalid yaml :")
    with pytest.raises(RuntimeError) as exc:
        load_config([path])
    assert "YAML syntax error" in str(exc.value)
    os.remove(path)

def test_load_config_invalid_root():
    fd, path = tempfile.mkstemp(suffix=".yaml")
    with os.fdopen(fd, "w") as f:
        f.write("- not_a_dict\n- still_not_a_dict")
    with pytest.raises(ValueError) as exc:
        load_config([path])
    assert "must contain a dictionary" in str(exc.value)
    os.remove(path)
