from __future__ import annotations

import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock
from pydantic import BaseModel
from ask.core.tools import ExecutorASK, ExecutorStateStore

@pytest.fixture
def state_store(tmp_path: Path) -> ExecutorStateStore:
    """Fixture for ExecutorStateStore."""
    return ExecutorStateStore(path=tmp_path / "test_state.yaml")

def test_executor_state_store_init(state_store: ExecutorStateStore, tmp_path: Path):
    """Test ExecutorStateStore initialization."""
    assert state_store.path == tmp_path / "test_state.yaml"
    assert state_store._data == {}

def test_executor_state_store_set_get(state_store: ExecutorStateStore):
    """Test ExecutorStateStore set and get methods."""
    state_store.set("key1", "value1")
    assert state_store.get("key1") == "value1"
    state_store.set("key2", {"a": 1, "b": 2})
    assert state_store.get("key2") == {"a": 1, "b": 2}
    assert state_store.get("non_existent_key") is None

def test_executor_state_store_persistence(tmp_path: Path):
    """Test that ExecutorStateStore persists data to a YAML file."""
    # ... existing test ...
    store1 = ExecutorStateStore(path=tmp_path / "test_state.yaml")
    store1.set("key1", "value1")
    store1.set("key2", [1, 2, 3])

    # Create a new instance to load from the same file
    store2 = ExecutorStateStore(path=tmp_path / "test_state.yaml")
    assert store2.get("key1") == "value1"
    assert store2.get("key2") == [1, 2, 3]
    assert store2.get("non_existent_key") is None

def test_executor_state_store_load_empty_file(tmp_path: Path):
    """Negative Test: Test loading from an empty YAML file."""
    file_path = tmp_path / "empty.yaml"
    file_path.touch()
    store = ExecutorStateStore(path=file_path)
    assert store._data == {}
    assert store.get("any_key") is None

def test_executor_state_store_load_invalid_yaml(tmp_path: Path):
    """Negative Test: Test loading from a file with invalid YAML."""
    file_path = tmp_path / "invalid.yaml"
    with open(file_path, "w") as f:
        f.write("key: - value: [")
    store = ExecutorStateStore(path=file_path)
    assert store._data == {}
    assert store.get("any_key") is None

@pytest.mark.asyncio
async def test_executor_step_with_different_inputs(state_store: ExecutorStateStore):
    """Positive Test: Ensure different inputs are cached separately."""
    executor: ExecutorASK[str, str] = ExecutorASK(store=state_store)
    mock_agent = MagicMock()
    mock_agent.run = AsyncMock(side_effect=["output1", "output2"])
    mock_agent._agent.output_type = str

    # First call with "input1"
    output1 = await executor.step(mock_agent, "input1")
    assert output1 == "output1"
    mock_agent.run.assert_called_once_with("input1")

    # Second call with "input2"
    output2 = await executor.step(mock_agent, "input2")
    assert output2 == "output2"
    mock_agent.run.assert_called_with("input2")
    assert mock_agent.run.call_count == 2

    # Third call with "input1" again, should be cached
    output3 = await executor.step(mock_agent, "input1")
    assert output3 == "output1"
    assert mock_agent.run.call_count == 2  # No new call to agent.run

@pytest.mark.asyncio
async def test_executor_step_with_corrupted_cache(state_store: ExecutorStateStore):
    """Negative Test: Test behavior with corrupted Pydantic data in cache."""
    class InputModel(BaseModel):
        value: str

    class OutputModel(BaseModel):
        result: str

    executor: ExecutorASK[InputModel, OutputModel] = ExecutorASK(store=state_store)
    
    # Manually insert corrupted data into the store
    input_data = InputModel(value="test")
    key = executor._get_input_key(input_data)
    state_store.set(key, {"wrong_field": "some_value"})

    mock_agent = MagicMock()
    mock_agent.run = AsyncMock(return_value=OutputModel(result="fresh_output"))
    mock_agent._agent.output_type = OutputModel

    # The executor should fail to validate the cached data and re-run the agent
    with pytest.raises(Exception):
        # Depending on Pydantic version, this could be ValidationError or other Exception
        await executor.step(mock_agent, input_data)

@pytest.mark.asyncio
async def test_executor_step_with_string_io(state_store: ExecutorStateStore):
    """Test Executor.step with string input and output."""
    executor: ExecutorASK[str, str] = ExecutorASK(store=state_store)
    mock_agent = MagicMock()
    mock_agent.run = AsyncMock(return_value="output_string")
    mock_agent._agent.output_type = str

    # First call, should call agent.run
    output1 = await executor.step(mock_agent, "input_string")
    assert output1 == "output_string"
    mock_agent.run.assert_called_once_with("input_string")

    # Second call, should use cache
    mock_agent.run.reset_mock()
    output2 = await executor.step(mock_agent, "input_string")
    assert output2 == "output_string"
    mock_agent.run.assert_not_called()


@pytest.mark.asyncio
async def test_executor_step_with_pydantic_io(state_store: ExecutorStateStore):
    """Test Executor.step with Pydantic model input and output."""
    class InputModel(BaseModel):
        value: str

    class OutputModel(BaseModel):
        result: str

    executor: ExecutorASK[InputModel, OutputModel] = ExecutorASK(store=state_store)
    mock_agent = MagicMock()
    mock_agent.run = AsyncMock(return_value=OutputModel(result="output_value"))
    mock_agent._agent.output_type = OutputModel

    input_data = InputModel(value="input_value")

    # First call, should call agent.run
    output1 = await executor.step(mock_agent, input_data)
    assert isinstance(output1, OutputModel)
    assert output1.result == "output_value"
    mock_agent.run.assert_called_once_with(input_data)

    # Second call, should use cache
    mock_agent.run.reset_mock()
    output2 = await executor.step(mock_agent, input_data)
    assert isinstance(output2, OutputModel)
    assert output2.result == "output_value"
    mock_agent.run.assert_not_called()
