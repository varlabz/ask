from __future__ import annotations

from pydantic import BaseModel


# some utils for base models
def example(model: BaseModel) -> str:
    """Get the example json for a pydantic model."""
    return model.model_dump_json(indent=2)


def schema(model: type[BaseModel]) -> dict:
    """Get the json schema for a pydantic model."""
    return model.model_json_schema().get("properties", {})


def load_string_json[Type: BaseModel](text: str, model: type[Type]) -> Type:
    """Load a pydantic model from a json string, stripping code block markers if present."""
    # strip ```json and ``` if present
    text = text.strip()
    if text.startswith("```json"):
        text = text[len("```json") :]
    elif text.startswith("```"):
        text = text[len("```") :]
    if text.endswith("```"):
        text = text[: -len("```")]
    return model.model_validate_json(text)
