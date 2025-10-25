import json

import pytest
from pydantic import BaseModel, ValidationError

from ask.core.context import example, load_string_json, schema


class SampleModel(BaseModel):
    name: str
    age: int


def test_example_positive():
    """Test example function with valid BaseModel."""
    model = SampleModel(name="test", age=25)
    result = example(model)
    assert isinstance(result, str)
    data = json.loads(result)
    assert data == {"name": "test", "age": 25}


def test_schema_positive():
    """Test schema function with valid model type."""
    result = schema(SampleModel)
    assert isinstance(result, dict)
    assert "properties" in result
    assert "name" in result["properties"]
    assert "age" in result["properties"]


def test_load_string_json_plain():
    """Test load_string_json with plain JSON string."""
    json_str = '{"name": "test", "age": 25}'
    result = load_string_json(json_str, SampleModel)
    assert result.name == "test"
    assert result.age == 25


def test_load_string_json_with_json_block():
    """Test load_string_json with JSON code block."""
    json_str = '```json\n{"name": "test", "age": 25}\n```'
    result = load_string_json(json_str, SampleModel)
    assert result.name == "test"
    assert result.age == 25


def test_load_string_json_with_block():
    """Test load_string_json with generic code block."""
    json_str = '```\n{"name": "test", "age": 25}\n```'
    result = load_string_json(json_str, SampleModel)
    assert result.name == "test"
    assert result.age == 25


def test_load_string_json_invalid_json():
    """Test load_string_json with invalid JSON (missing required field)."""
    with pytest.raises(ValidationError):
        load_string_json('{"name": "test"}', SampleModel)


def test_load_string_json_empty_string():
    """Test load_string_json with empty string."""
    with pytest.raises(ValidationError):
        load_string_json("", SampleModel)


def test_load_string_json_not_json():
    """Test load_string_json with non-JSON string."""
    with pytest.raises(ValidationError):
        load_string_json("not a json string", SampleModel)
