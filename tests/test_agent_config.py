import pytest
from pydantic import ValidationError

from ask.core.config import AgentConfig


class TestAgentConfig:
    def test_builtin_type_str(self):
        config = AgentConfig(instructions="test", output_type="str")
        assert config.output_type is str

    def test_builtin_type_int(self):
        config = AgentConfig(instructions="test", output_type="int")
        assert config.output_type is int

    def test_builtin_type_float(self):
        config = AgentConfig(instructions="test", output_type="float")
        assert config.output_type is float

    def test_builtin_type_bool(self):
        config = AgentConfig(instructions="test", output_type="bool")
        assert config.output_type is bool

    def test_type_object(self):
        config = AgentConfig(instructions="test", output_type=dict)
        assert config.output_type is dict

    def test_type_list_str(self):
        config = AgentConfig(instructions="test", output_type="list[str]")
        assert config.output_type == list[str]

    def test_default_type(self):
        config = AgentConfig(instructions="test")
        assert config.output_type is str

    def test_invalid_type_raises(self):
        with pytest.raises(
            ValidationError, match="Unknown or invalid type string: invalid_type"
        ):
            AgentConfig(instructions="test", output_type="invalid_type")
