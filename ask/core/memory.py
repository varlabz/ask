from pydantic_ai import ModelMessage
from pydantic_ai.messages import (
    ModelMessagesTypeAdapter,
    RetryPromptPart,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_core import to_json

from ask.core.config import LLMConfig


class Memory:
    """Basic in-memory message history storage."""

    _history: list[ModelMessage] = []
    _next: "Memory | None" = None

    def __init__(self, next: "Memory | None" = None):
        self._next = next
        if next is not None:
            self._history = next.get()

    def get(self) -> list[ModelMessage]:
        return self._history

    def set(self, messages: list[ModelMessage]):
        self._history = messages
        if self._next is not None:
            self._next.set(messages)


class NoMemory(Memory):
    """Memory implementation that does not store any messages."""

    def get(self) -> list:
        return []

    def set(self, messages: list):
        pass


class MemoryToolsCompression(Memory):
    """Memory implementation that compresses tool messages."""

    def set(self, messages: list[ModelMessage]):
        super().set(_repack_tools_messages(messages))


class FileMemory(Memory):
    """File-based memory implementation."""

    def __init__(self, file_path: str):
        self._file_path = file_path
        self._history = self.load_from_file(file_path)

    def set(self, messages: list[ModelMessage]):
        self.save_to_file(self._file_path, messages)
        super().set(messages)

    @staticmethod
    def save_to_file(file_path: str, history: list[ModelMessage]) -> None:
        as_json = to_json(history)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(as_json.decode("utf-8"))

    @staticmethod
    def load_from_file(file_path: str) -> list[ModelMessage]:
        try:
            with open(file_path, encoding="utf-8") as f:
                json_str = f.read()
            return ModelMessagesTypeAdapter.validate_json(json_str.encode("utf-8"))
        except FileNotFoundError:
            return []


def memory_factory(llm: LLMConfig, file_path: str | None) -> Memory:
    """Create a Memory instance based on LLMConfig and optional file path."""
    if file_path is not None:
        return (
            MemoryToolsCompression(FileMemory(file_path))
            if llm.compress_history
            else Memory(FileMemory(file_path))
        )
    else:
        return MemoryToolsCompression() if llm.compress_history else Memory()


def _repack_tools_messages(messages: list[ModelMessage]) -> list[ModelMessage]:
    """
    Remove tool calls and tool responses; keep surrounding conversation intact.

    Rules:
    - Drop any message whose parts are exclusively tool traffic:
      * ToolCallPart (model decided to call a tool)
      * ToolReturnPart (tool result returned to the model)
      * RetryPromptPart (tool error/retry prompt)
    - Keep all other messages unchanged.
    """
    tool_parts = (ToolCallPart, ToolReturnPart, RetryPromptPart)

    def is_tool_only_message(msg: ModelMessage) -> bool:
        parts = msg.parts
        if not parts:
            return False

        return any(isinstance(p, tool_parts) for p in parts)

    # dump_messages(messages, "tmp/message_dump.txt")
    # return dump_messages([m for m in messages if not is_tool_only_message(m)], "tmp/message_dump_filtered.txt")
    return [m for m in messages if not is_tool_only_message(m)]
