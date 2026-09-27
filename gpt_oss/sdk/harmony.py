"""Engine for in-process, token-level backends (triton, torch, vllm, metal, ...).

The engine renders the conversation in the harmony format, streams tokens
from a generator, parses them, and runs tool calls until the model answers.
"""

from __future__ import annotations

import dataclasses
import datetime
import itertools
import json
import threading
import uuid
from collections.abc import Callable, Iterable, Iterator
from typing import Any, Protocol

from openai_harmony import (
    Author,
    Conversation,
    DeveloperContent,
    HarmonyEncoding,
    HarmonyEncodingName,
    Message,
    ReasoningEffort,
    Role,
    StreamableParser,
    StreamState,
    SystemContent,
    ToolDescription,
    load_harmony_encoding,
)

from .engine import ChatConfig, Engine, call_function
from .tools import run_awaitable
from .types import Event, Reply, ToolCall, Usage

_EFFORT = {
    "low": ReasoningEffort.LOW,
    "medium": ReasoningEffort.MEDIUM,
    "high": ReasoningEffort.HIGH,
}


class TokenGenerator(Protocol):
    """What `HarmonyEngine` needs from a backend.

    Matches the `TokenGenerator` classes in `gpt_oss.torch`, `gpt_oss.triton`
    and `gpt_oss.vllm`.
    """

    def generate(
        self,
        prompt_tokens: list[int],
        stop_tokens: list[int],
        temperature: float = 1.0,
        max_tokens: int = 0,
    ) -> Iterable[int]: ...


class NextTokenGenerator:
    """Adapt an ``infer_next_token(tokens, temperature, new_request) -> int``
    function (the `gpt_oss.responses_api.inference` backends) to `TokenGenerator`."""

    def __init__(self, infer_next_token: Callable[[list[int], float, bool], int]):
        self.infer_next_token = infer_next_token

    def generate(
        self,
        prompt_tokens: list[int],
        stop_tokens: list[int],
        temperature: float = 1.0,
        max_tokens: int = 0,
    ) -> Iterator[int]:
        tokens = list(prompt_tokens)
        new_request = True
        while max_tokens == 0 or len(tokens) - len(prompt_tokens) < max_tokens:
            token = self.infer_next_token(tokens, temperature, new_request)
            new_request = False
            tokens.append(token)
            yield token
            if token in stop_tokens:
                return


class HarmonyEngine(Engine):
    def __init__(self, generator: TokenGenerator, encoding: HarmonyEncoding | None = None):
        self.generator = generator
        self.encoding = encoding or load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
        self._stop_tokens = list(self.encoding.stop_tokens_for_assistant_actions())
        # Local models keep a single KV cache, so only one generation may run at a time.
        self._lock = threading.Lock()

    def render(self, history: list[Message], config: ChatConfig) -> list[int]:
        system = SystemContent.new().with_reasoning_effort(_EFFORT[config.reasoning_effort])
        system = system.with_conversation_start_date(
            config.start_date or datetime.date.today().isoformat()
        )
        for builtin in config.builtin_tools:
            system = system.with_tools(builtin.tool_config)
        messages = [Message.from_role_and_content(Role.SYSTEM, system)]
        if config.instructions or config.tools:
            developer = DeveloperContent.new()
            if config.instructions:
                developer = developer.with_instructions(config.instructions)
            if config.tools:
                developer = developer.with_function_tools(
                    [ToolDescription.new(t.name, t.description, parameters=t.parameters) for t in config.tools]
                )
            messages.append(Message.from_role_and_content(Role.DEVELOPER, developer))
        conversation = Conversation.from_messages(messages + history)
        return self.encoding.render_conversation_for_completion(conversation, Role.ASSISTANT)

    def run(self, history: list[Message], message: str, config: ChatConfig) -> Iterator[Event]:
        history.append(Message.from_role_and_content(Role.USER, message))
        builtins = {b.name: b for b in config.builtin_tools}
        text: list[str] = []
        reasoning: list[str] = []
        calls: list[ToolCall] = []
        usage = Usage()
        finish_reason = "stop"

        for tool_round in itertools.count():
            prompt = self.render(history, config)
            usage.prompt_tokens = len(prompt)
            parser = StreamableParser(self.encoding, role=Role.ASSISTANT)
            generated = 0
            with self._lock:
                tokens = self.generator.generate(
                    prompt,
                    stop_tokens=self._stop_tokens,
                    temperature=config.temperature,
                    max_tokens=config.max_output_tokens or 0,
                )
                try:
                    for token in tokens:
                        generated += 1
                        parser.process(token)
                        delta = parser.last_content_delta
                        if delta and parser.current_recipient is None:
                            if parser.current_channel == "final":
                                yield Event("text", text=delta)
                            elif parser.current_channel == "analysis":
                                yield Event("reasoning", text=delta)
                        if token in self._stop_tokens:
                            break
                finally:
                    if hasattr(tokens, "close"):
                        tokens.close()
            usage.completion_tokens += generated

            new_messages = list(parser.messages)
            truncated = parser.state != StreamState.EXPECT_START
            if truncated and parser.current_content and parser.current_recipient is None:
                partial = Message.from_role_and_content(Role.ASSISTANT, parser.current_content)
                new_messages.append(partial.with_channel(parser.current_channel or "final"))
            history.extend(new_messages)

            for m in new_messages:
                if m.recipient is None and m.channel == "final":
                    text.append(_text_of(m))
                elif m.recipient is None and m.channel == "analysis":
                    reasoning.append(_text_of(m))

            last = new_messages[-1] if new_messages else None
            if truncated or last is None or last.recipient in (None, "assistant"):
                hit_limit = config.max_output_tokens and generated >= config.max_output_tokens
                finish_reason = "length" if hit_limit else "stop"
                break

            if tool_round >= config.max_tool_rounds:
                history.append(_tool_message(last, "Error: tool call limit reached"))
                finish_reason = "max_tool_rounds"
                break
            call = ToolCall(
                id=f"call_{uuid.uuid4().hex}",
                name=last.recipient.removeprefix("functions."),
                arguments=_text_of(last),
            )
            yield Event("tool_call", tool_call=call)
            results = self._run_tool(last, call, builtins, config)
            history.extend(results)
            call = dataclasses.replace(call, output="\n".join(_text_of(m) for m in results))
            calls.append(call)
            yield Event("tool_result", tool_call=call)

        reply = Reply(
            text="\n".join(text),
            reasoning="\n".join(reasoning),
            tool_calls=calls,
            finish_reason=finish_reason,
            usage=usage,
        )
        yield Event("done", reply=reply)

    def _run_tool(
        self, message: Message, call: ToolCall, builtins: dict[str, Any], config: ChatConfig
    ) -> list[Message]:
        recipient = message.recipient
        if recipient.startswith("functions."):
            output = call_function(config, call.name, call.arguments)
        else:
            builtin = builtins.get(recipient.split(".", 1)[0])
            if builtin is not None:
                return run_awaitable(_collect(builtin.process(message)))
            output = f"Error: unknown tool {recipient!r}"
        return [_tool_message(message, output)]

    def dump_history(self, history: list[Message]) -> list[dict[str, Any]]:
        return [json.loads(m.to_json()) for m in history]

    def load_history(self, data: list[dict[str, Any]]) -> list[Message]:
        return [Message.from_dict(m) for m in data]


def _tool_message(call: Message, output: str) -> Message:
    result = Message.from_author_and_content(Author.new(Role.TOOL, call.recipient), output)
    return result.with_recipient("assistant").with_channel(call.channel or "commentary")


async def _collect(messages: Any) -> list[Message]:
    return [m async for m in messages]


def _text_of(message: Message) -> str:
    return "".join(getattr(c, "text", "") for c in message.content)
