"""The interface every SDK backend implements."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, Literal

from .tools import FunctionTool
from .types import Event

ReasoningEffort = Literal["low", "medium", "high"]


@dataclass
class ChatConfig:
    instructions: str | None = None
    tools: list[FunctionTool] = field(default_factory=list)
    # Harmony-native tools (`gpt_oss.tools.tool.Tool` instances such as the
    # browser or python tool). Only local backends can run these.
    builtin_tools: list[Any] = field(default_factory=list)
    reasoning_effort: ReasoningEffort = "low"
    temperature: float = 1.0
    max_output_tokens: int | None = None
    max_tool_rounds: int = 8
    start_date: str | None = None

    def function(self, name: str) -> FunctionTool | None:
        return next((t for t in self.tools if t.name == name), None)


class Engine(ABC):
    """Runs one user turn, including any tool calls, against some backend.

    `run` appends the new messages to `history` (whose element type is owned
    by the engine) and yields `Event`s, ending with a ``done`` event.
    """

    @abstractmethod
    def run(self, history: list[Any], message: str, config: ChatConfig) -> Iterator[Event]:
        ...

    def dump_history(self, history: list[Any]) -> list[dict[str, Any]]:
        """Convert `history` to JSON-serializable dicts."""
        return [dict(m) for m in history]

    def load_history(self, data: list[dict[str, Any]]) -> list[Any]:
        """Inverse of `dump_history`."""
        return [dict(m) for m in data]


def call_function(config: ChatConfig, name: str, arguments: str) -> str:
    fn = config.function(name)
    if fn is None:
        return f"Error: unknown function {name!r}"
    return fn.invoke(arguments)
