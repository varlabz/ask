from __future__ import annotations

import hashlib
import json
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import (
    Any,
    Protocol,
    runtime_checkable,
)

import yaml
from pydantic import BaseModel


@runtime_checkable
class CacheStore(Protocol):
    """
    Interface for a simple key-value cache store used by CacheASK.

    Implementations must be persistent or in-memory key-value stores
    that support basic get/set and cleanup operations.
    """

    def get(self, key: str) -> Any | None:
        """Return value for key or None if not found."""
        ...

    def set(self, key: str, value: Any) -> None:
        """Store value under key."""
        ...

    def clean(self) -> None:
        """Clear underlying storage (and remove backing file if any)."""
        ...


class CacheStoreYaml(CacheStore):
    """
    YAML file-backed implementation of CacheStore for caching
    executor/agent step inputs and outputs.
    """

    def __init__(self, path: str | Path = ".ask_cache.yaml"):
        self.path = Path(path).expanduser().resolve()
        self._data = self._load()

    def _load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {}
        with open(self.path) as f:
            try:
                data = yaml.safe_load(f)
                return data if isinstance(data, dict) else {}
            except yaml.YAMLError:
                # Handle empty or invalid YAML file
                return {}

    def _save(self):
        with open(self.path, "w") as f:
            yaml.dump(self._data, f, default_flow_style=False)

    def get(self, key: str) -> Any | None:
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


class CacheStoreMemory(CacheStore):
    """
    In-memory implementation of CacheStore. Useful for tests or ephemeral runs.
    """

    def __init__(self):
        self._data: dict[str, Any] = {}

    def get(self, key: str) -> Any | None:
        return self._data.get(key)

    def set(self, key: str, value: Any) -> None:
        self._data[key] = value

    def clean(self) -> None:
        self._data.clear()


class CacheASK:
    def __init__(self, store: CacheStore | None = None):
        self.store: CacheStore = store or CacheStoreYaml()

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

        return hashlib.sha256(serialized_input.encode("utf-8")).hexdigest()

    @asynccontextmanager
    async def step[OutputT](
        self, input: Any
    ) -> AsyncIterator[tuple[OutputT | None, Callable[[OutputT], Any]]]:
        """
        Async context manager for executing a step.
        """
        key = self._get_input_key(input)
        output = self.store.get(key)

        def set_output(x: OutputT) -> OutputT:
            if isinstance(x, BaseModel):
                self.store.set(key, x.model_dump())
            else:
                self.store.set(key, x)
            return x

        try:
            yield (output, set_output)
        finally:
            pass

    def clean(self):
        """
        Clean the executor's state store.
        """
        self.store.clean()
