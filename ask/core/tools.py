from __future__ import annotations

import hashlib
import json
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Generic, List, Optional, cast
from xml.sax.saxutils import escape

import yaml
from pydantic import BaseModel, Field

from ask.core.agent import AgentASK, InputT, OutputT

def _pydantic_model_to_xml(model: type[BaseModel]) -> str:
    def _value(tag: str, field: Any) -> str:
        if field.description:
            return f'<{tag}>{escape(str(field.description))}</{tag}>'
        # if field has annotations, convert to human readable type
        if field.annotation:
            type_map = {
                str: 'string',
                int: 'number',
                float: 'number',
                bool: 'boolean',
            }
            return f'<{tag}>{escape(str(type_map.get(field.annotation, "unknown")))}</{tag}>'
        if field is None:
            return ""
        return f'<{tag}>{escape(str(field))}</{tag}>'

    def _model(mod: type[BaseModel]) -> str:
        return '\n'.join(_value(k, v) for k, v in mod.model_fields.items())

    return _model(model)

def _model_instance_to_xml(instance: BaseModel) -> str:
    """Serialize a Pydantic model instance to XML string."""
    def _value(tag: str, val: Any) -> str:
        if isinstance(val, BaseModel):
            return _model(val)
        if isinstance(val, Enum):
            return f'<{tag}>{escape(str(val.value))}</{tag}>'
        if isinstance(val, (list, tuple)):
            return "".join(_value(tag, i) for i in val)
        if isinstance(val, dict):
            import json
            return f'<{tag}>{escape(json.dumps(val, ensure_ascii=False))}</{tag}>'
        if val is None:
            return ""
        return f'<{tag}>{escape(str(val))}</{tag}>'

    def _model(mod: BaseModel) -> str:
        return '\n'.join(_value(k, getattr(mod, k)) for k, v in type(mod).model_fields.items())

    return _model(instance)

class ContextASK(BaseModel):
    # input schema for parameters
    @classmethod
    def to_input(cls) -> str:
        return _pydantic_model_to_xml(cls)

    # output parameter values
    def to_output(self) -> str:
        return _model_instance_to_xml(self)

    # make str() because it is more natural to use str() to get the output for prompt formatting
    def __str__(self) -> str:
        return self.to_output()
    
import yaml

class ExecutorStateStore:
    """
    Persistent storage for executor step inputs/outputs.
    """

    def __init__(self, path: str | Path = ".ask_executor_state.yaml"):
        self.path = Path(path).expanduser().resolve()
        self._data = self._load()

    def _load(self) -> Dict[str, Any]:
        if not self.path.exists():
            return {}
        with open(self.path, "r") as f:
            try:
                data = yaml.safe_load(f)
                return data if isinstance(data, dict) else {}
            except yaml.YAMLError:
                # Handle empty or invalid YAML file
                return {}

    def _save(self):
        with open(self.path, "w") as f:
            yaml.dump(self._data, f, default_flow_style=False)

    def get(self, key: str) -> Optional[Any]:
        """
        Get value by key. Returns None if key does not exist.
        """
        return self._data.get(key)

    def set(self, key: str, value: Any):
        """
        Store key and value.
        """
        self._data[key] = value
        self._save()

    def clean(self):
        """
        Clear the storage by deleting the state file and clearing the in-memory data.
        """
        self._data.clear()
        if self.path.exists():
            self.path.unlink()

class ExecutorASK:
    """
    Executes agent steps with a pluggable state store, caching results.
    """

    def __init__(self, store: ExecutorStateStore | None = None):
        self.store = store or ExecutorStateStore()

    def _get_input_key(self, input_data: Any) -> str:
        """Create a consistent hash key from the input data."""
        if isinstance(input_data, BaseModel):
            serialized_input = json.dumps(input_data.model_dump(), sort_keys=True)
        elif isinstance(input_data, str):
            serialized_input = input_data
        else:
            try:
                serialized_input = json.dumps(input_data, sort_keys=True)
            except TypeError:
                serialized_input = str(input_data)
        
        return hashlib.sha256(serialized_input.encode('utf-8')).hexdigest()

    async def step(self, agent: AgentASK[InputT, OutputT], input_data: InputT) -> OutputT:
        """
        Execute a step, using the cache if possible.
        """
        key = self._get_input_key(input_data)
        cached_output = self.store.get(key)
        
        if cached_output is not None:
            # If the output type is a Pydantic model, we need to reconstruct it
            output_type = agent._agent.output_type
            if isinstance(output_type, type) and issubclass(output_type, BaseModel):
                return cast(OutputT, output_type.model_validate(cached_output))
            return cast(OutputT, cached_output)

        output = await agent.run(input_data)
        
        # If the output is a Pydantic model, store its dictionary representation
        if isinstance(output, BaseModel):
            self.store.set(key, output.model_dump())
        else:
            self.store.set(key, output)
            
        return output

    def clean(self):
        """
        Clean the executor's state store.
        """
        self.store.clean()


if __name__ == "__main__":
    class Param(ContextASK):
        title: str = Field(..., description="Title of the item")
        count: int
        var: Any

    param = Param(title="Example", count=5, var={"key": "value"})

    print(Param.to_input())

    print(param.to_output())
