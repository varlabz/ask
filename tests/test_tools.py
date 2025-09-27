from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import BaseModel, ValidationError

from ask.core.agent_cache import CacheASK, CacheStoreYaml


@pytest.fixture
def state_store(tmp_path: Path) -> CacheStoreYaml:
    """Fixture for CacheStateStore."""
    return CacheStoreYaml(path=tmp_path / "test_state.yaml")


def test_cache_state_store_init(state_store: CacheStoreYaml, tmp_path: Path):
    """Test ExecutorStateStore initialization."""
    assert state_store.path == tmp_path / "test_state.yaml"
    assert state_store._data == {}


def test_cache_state_store_set_get(state_store: CacheStoreYaml):
    """Test ExecutorStateStore set and get methods."""
    state_store.set("key1", "value1")
    assert state_store.get("key1") == "value1"
    state_store.set("key2", {"a": 1, "b": 2})
    assert state_store.get("key2") == {"a": 1, "b": 2}
    assert state_store.get("non_existent_key") is None


def test_cache_state_store_persistence(tmp_path: Path):
    """Test that ExecutorStateStore persists data to a YAML file."""
    # ... existing test ...
    store1 = CacheStoreYaml(path=tmp_path / "test_state.yaml")
    store1.set("key1", "value1")
    store1.set("key2", [1, 2, 3])

    # Create a new instance to load from the same file
    store2 = CacheStoreYaml(path=tmp_path / "test_state.yaml")
    assert store2.get("key1") == "value1"
    assert store2.get("key2") == [1, 2, 3]
    assert store2.get("non_existent_key") is None


def test_cache_state_store_load_empty_file(tmp_path: Path):
    """Negative Test: Test loading from an empty YAML file."""
    file_path = tmp_path / "empty.yaml"
    file_path.touch()
    store = CacheStoreYaml(path=file_path)
    assert store._data == {}
    assert store.get("any_key") is None


def test_cache_state_store_load_invalid_yaml(tmp_path: Path):
    """Negative Test: Test loading from a file with invalid YAML."""
    file_path = tmp_path / "invalid.yaml"
    with open(file_path, "w") as f:
        f.write("key: - value: [")
    store = CacheStoreYaml(path=file_path)
    assert store._data == {}
    assert store.get("any_key") is None


@pytest.mark.asyncio
async def test_cache_step_with_different_inputs(state_store: CacheStoreYaml):
    """Positive Test: Ensure different inputs are cached separately."""
    executor = CacheASK(store=state_store)
    mock_agent = MagicMock()
    mock_agent.run = AsyncMock(side_effect=["output1", "output2"])

    # First call with input1
    async with executor.step("input1") as (cached, set_output):
        if cached is None:
            out = await mock_agent.run("input1")
            set_output(out)
            output1 = out
        else:
            output1 = cached
    assert output1 == "output1"
    mock_agent.run.assert_called_once_with("input1")

    # Second call with input2
    async with executor.step("input2") as (cached, set_output):
        if cached is None:
            out = await mock_agent.run("input2")
            set_output(out)
            output2 = out
        else:
            output2 = cached
    assert output2 == "output2"
    mock_agent.run.assert_called_with("input2")
    assert mock_agent.run.call_count == 2

    # Third call with input1 should hit cache
    async with executor.step("input1") as (cached, _):
        output3 = cached
    assert output3 == "output1"
    assert mock_agent.run.call_count == 2


@pytest.mark.asyncio
async def test_cache_step_with_corrupted_cache(state_store: CacheStoreYaml):
    class InputModel(BaseModel):
        value: str

    class OutputModel(BaseModel):
        result: str

    executor = CacheASK(store=state_store)
    input_data = InputModel(value="test")
    key = executor._get_input_key(input_data)
    state_store.set(key, {"wrong_field": "some_value"})

    with pytest.raises(ValidationError):
        async with executor.step(input_data) as (cached, _):
            assert cached is not None
            OutputModel.model_validate(cached)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_cache_step_with_string_io(state_store: CacheStoreYaml):
    executor = CacheASK(store=state_store)
    mock_agent = MagicMock()
    mock_agent.run = AsyncMock(return_value="output_string")

    async with executor.step("input_string") as (cached, set_output):
        if cached is None:
            out = await mock_agent.run("input_string")
            set_output(out)
            output1 = out
        else:
            output1 = cached
    assert output1 == "output_string"
    mock_agent.run.assert_called_once_with("input_string")

    mock_agent.run.reset_mock()
    async with executor.step("input_string") as (cached, _):
        output2 = cached
    assert output2 == "output_string"
    mock_agent.run.assert_not_called()


@pytest.mark.asyncio
async def test_cache_step_with_pydantic_io(state_store: CacheStoreYaml):
    class InputModel(BaseModel):
        value: str

    class OutputModel(BaseModel):
        result: str

    executor = CacheASK(store=state_store)
    mock_agent = MagicMock()
    mock_agent.run = AsyncMock(return_value=OutputModel(result="output_value"))

    input_data = InputModel(value="input_value")

    async with executor.step(input_data) as (cached, set_output):
        if cached is None:
            out = await mock_agent.run(input_data)
            set_output(out)
            output1 = out
        else:
            output1 = OutputModel.model_validate(cached)  # type: ignore[arg-type]
    assert isinstance(output1, OutputModel)
    assert output1.result == "output_value"
    mock_agent.run.assert_called_once_with(input_data)

    mock_agent.run.reset_mock()
    async with executor.step(input_data) as (cached, _):
        assert cached is not None
        output2 = OutputModel.model_validate(cached)  # type: ignore[arg-type]
    assert isinstance(output2, OutputModel)
    assert output2.result == "output_value"
    mock_agent.run.assert_not_called()
