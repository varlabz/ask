from __future__ import annotations

from enum import Enum
from typing import Any, Dict, Generic, List, Optional
import json
import hashlib
from pathlib import Path
from xml.sax.saxutils import escape
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
    
class ExecutorStateStore:
    """Persistent storage for executor step inputs/outputs.

    Stores JSON lines with fields: key, input, output.
    """

    def __init__(self, path: str | Path = ".ask_executor_state.jsonl"):
        self.path = Path(path).expanduser().resolve()

class Executor(Generic[InputT, OutputT]):
    """Executes agent steps with a pluggable state store."""

    def __init__(self, store: Optional[ExecutorStateStore] = None):
        self.store = store or ExecutorStateStore()

    async def step(self, agent: AgentASK[InputT, OutputT], input_data: InputT) -> OutputT:
        pass
    
    
if __name__ == "__main__":
    class Param(ContextASK):
        title: str = Field(..., description="Title of the item")
        count: int
        var: Any

    param = Param(title="Example", count=5, var={"key": "value"})

    print(Param.to_input())

    print(param.to_output())
