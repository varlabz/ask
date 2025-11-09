from collections.abc import Callable
from typing import Any

from pydantic_ai.messages import ModelResponse, TextPart
from pydantic_ai.models import Model
from pydantic_ai.models.wrapper import WrapperModel


class ThoughtWrapper(WrapperModel):
    def __init__(self, wrapped: Model, on_before_tool_use: Callable[[str], str | None]):
        super().__init__(wrapped)
        self._on_before_tool_use = on_before_tool_use

    async def request(self, *args: Any, **kwargs: Any) -> ModelResponse:
        response = await self.wrapped.request(*args, **kwargs)
        # print(f"#  {args} // {kwargs}", file=sys.stderr)
        # print(f"### Tool calls {response.parts}", file=sys.stderr)
        # try:
        #     if (
        #         response.parts
        #         and isinstance(response.parts[0], TextPart)
        #         and response.parts[0].content
        #     ):
        #         json.loads(response.parts[0].content)
        # except json.JSONDecodeError as e:
        #     print(f">>>> JSON decode error: {e}", file=sys.stderr)
        # print("-------------------------------------------------------------------------", file=sys.stderr)
        if response.tool_calls:
            new_parts = list(response.parts)
            modified = False
            for i, part in enumerate(response.parts):
                if isinstance(part, TextPart) and part.content:
                    new_content = self._on_before_tool_use(part.content)
                    if new_content is not None:
                        new_parts[i] = TextPart(content=new_content)
                        modified = True
                    break
            if modified:
                return ModelResponse(
                    parts=new_parts,
                    usage=response.usage,
                    model_name=response.model_name,
                    timestamp=response.timestamp,
                )
        return response
