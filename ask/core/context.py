from __future__ import annotations

from collections.abc import Generator
from enum import Enum
from typing import Any
from xml.sax.saxutils import escape

from pydantic import BaseModel, Field


def _pydantic_model_to_xml(model: type[ContextASK]) -> str:
    """Serialize a Pydantic model class to XML schema string."""

    def _tag(
        tag: str,
        field: Any,
    ) -> str:
        if not field.description:
            raise ValueError(f"Field {tag} must have a description")
        attrs = f"description='{escape(str(field.description))}'"
        if not field.is_required():
            default_val = (
                field.default
                if field.default_factory is None
                else field.default_factory()
            )
            attrs += f" default='{escape(str(default_val))}'"
        return f"<{tag} {attrs}>"

    def _value(tag: str, field: Any, intend: int = 0) -> Generator[str, Any, Any]:
        # Check if field.annotation is a ContextASK subclass
        if isinstance(field.annotation, type) and issubclass(
            field.annotation, ContextASK
        ):
            yield f"{_tag(tag, field)}\n"
            yield f"{_model(field.annotation, intend + 1)}\n"
            yield f"</{tag}>"
        else:
            yield f"{_tag(tag, field)}</{tag}>"

    def _model(mod: type[ContextASK], intend: int = 0) -> str:
        return "\n".join(
            f"{' ' * intend}{''.join(_value(k, v, intend))}"
            for k, v in mod.model_fields.items()
        )

    return _model(model)


def _model_instance_to_xml(instance: ContextASK) -> str:
    """Serialize a Pydantic model instance to XML string."""

    def _value(tag: str, val: Any, intend: int = 0) -> str:
        if isinstance(val, ContextASK):
            return f"<{tag}>\n{' ' * (intend + 1)}{_model(val, intend + 1)}\n</{tag}>"
        if isinstance(val, Enum):
            return f"<{tag}>{escape(str(val.value))}</{tag}>"
        if isinstance(val, (list, tuple)):
            return "\n".join(_value(tag, i) for i in val)
        if isinstance(val, dict):
            import json

            return f"<{tag}>{escape(json.dumps(val, ensure_ascii=False))}</{tag}>"
        if val is None:
            return ""
        return f"<{tag}>{escape(str(val))}</{tag}>"

    def _model(mod: BaseModel, intend: int = 0) -> str:
        return "\n".join(
            _value(k, getattr(mod, k)) for k, v in type(mod).model_fields.items()
        )

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


if __name__ == "__main__":

    class Param1(ContextASK):
        class Param(ContextASK):
            val: str = Field(..., description="A string value")

        title: str = Field("123", description="Title of the item")
        count: int = Field(..., description="Count of items")
        var: Any = Field(..., description="A variable of any type")
        param: Param = Field(..., description="A nested parameter")
        lst: list[str] = Field(..., description="A list of tags")
        list_def: list[str] = Field(default_factory=list, description="A list of tags")

    param = Param1(
        title="Example",
        count=5,
        var={"key": "value", "key2": "value2"},
        lst=["tag1", "tag2"],
        param=Param1.Param(val="Nested++"),
    )

    print(Param1.to_input())
    print("--------")
    print(param.to_output())
