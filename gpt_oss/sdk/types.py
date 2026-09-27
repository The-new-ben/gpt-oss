"""Plain data types returned by the SDK."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Literal


class GptOssError(RuntimeError):
    """Raised when a backend fails or returns something the SDK cannot use."""


@dataclass(frozen=True)
class ToolCall:
    """A tool call made by the model while producing a reply."""

    id: str
    name: str
    arguments: str
    output: str | None = None

    @property
    def args(self) -> dict[str, Any]:
        """The call arguments decoded from JSON (empty dict if they are not a JSON object)."""
        try:
            value = json.loads(self.arguments) if self.arguments.strip() else {}
        except json.JSONDecodeError:
            return {}
        return value if isinstance(value, dict) else {}


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0


@dataclass
class Reply:
    """The result of one `Chat.send` call."""

    text: str
    reasoning: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    # "stop", "length" (hit max_output_tokens) or "max_tool_rounds"
    finish_reason: str = "stop"
    usage: Usage = field(default_factory=Usage)

    def __str__(self) -> str:
        return self.text


EventType = Literal["reasoning", "text", "tool_call", "tool_result", "done"]


@dataclass(frozen=True)
class Event:
    """A streaming event.

    - ``reasoning`` / ``text``: ``text`` holds the new chunk.
    - ``tool_call``: ``tool_call`` is about to run (``output`` is None).
    - ``tool_result``: ``tool_call`` finished (``output`` is set).
    - ``done``: ``reply`` holds the complete `Reply`. Always the last event.
    """

    type: EventType
    text: str = ""
    tool_call: ToolCall | None = None
    reply: Reply | None = None
