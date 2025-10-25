from __future__ import annotations

from pydantic import BaseModel


# some utils for base models
def example(model: BaseModel):
    return model.model_dump_json(indent=2)
