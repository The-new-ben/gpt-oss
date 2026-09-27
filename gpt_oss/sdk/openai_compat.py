"""Engine for any server with an OpenAI-compatible chat completions API.

Works with Ollama, vLLM (`vllm serve`), LM Studio, llama.cpp and hosted
providers serving gpt-oss. The server applies the harmony format itself.
"""

from __future__ import annotations

import dataclasses
import itertools
import json
import uuid
from collections.abc import Iterator
from typing import Any

import requests

from .engine import ChatConfig, Engine, call_function
from .types import Event, GptOssError, Reply, ToolCall, Usage


class OpenAICompatibleEngine(Engine):
    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        api_key: str | None = None,
        timeout: float = 600.0,
        extra_body: dict[str, Any] | None = None,
        session: requests.Session | None = None,
    ):
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model = model
        self.timeout = timeout
        self.extra_body = extra_body or {}
        self.session = session or requests.Session()
        self.headers = {"Content-Type": "application/json"}
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"

    def run(self, history: list[dict[str, Any]], message: str, config: ChatConfig) -> Iterator[Event]:
        if config.builtin_tools:
            raise GptOssError(
                "Built-in harmony tools (browser, python) need a local backend; "
                "pass plain functions as tools instead."
            )
        history.append({"role": "user", "content": message})
        text: list[str] = []
        reasoning: list[str] = []
        calls: list[ToolCall] = []
        usage = Usage()
        finish_reason = "stop"

        for tool_round in itertools.count():
            round_text, round_reasoning = "", ""
            pending: list[dict[str, str]] = []
            round_finish = None
            for chunk in self._stream(history, config):
                if chunk.get("usage"):
                    usage.prompt_tokens = chunk["usage"].get("prompt_tokens", 0)
                    usage.completion_tokens += chunk["usage"].get("completion_tokens", 0)
                for choice in chunk.get("choices") or []:
                    delta = choice.get("delta") or {}
                    thought = delta.get("reasoning_content") or delta.get("reasoning")
                    if thought:
                        round_reasoning += thought
                        yield Event("reasoning", text=thought)
                    if delta.get("content"):
                        round_text += delta["content"]
                        yield Event("text", text=delta["content"])
                    for part in delta.get("tool_calls") or []:
                        _merge_tool_call_delta(pending, part)
                    round_finish = choice.get("finish_reason") or round_finish

            if round_text:
                text.append(round_text)
            if round_reasoning:
                reasoning.append(round_reasoning)
            assistant: dict[str, Any] = {"role": "assistant", "content": round_text}
            if pending:
                assistant["tool_calls"] = [
                    {
                        "id": p["id"],
                        "type": "function",
                        "function": {"name": p["name"], "arguments": p["arguments"]},
                    }
                    for p in pending
                ]
            history.append(assistant)

            if not pending:
                finish_reason = "length" if round_finish == "length" else "stop"
                break
            if tool_round >= config.max_tool_rounds:
                for p in pending:
                    history.append(
                        {"role": "tool", "tool_call_id": p["id"], "content": "Error: tool call limit reached"}
                    )
                finish_reason = "max_tool_rounds"
                break
            for p in pending:
                call = ToolCall(id=p["id"], name=p["name"], arguments=p["arguments"])
                yield Event("tool_call", tool_call=call)
                output = call_function(config, call.name, call.arguments)
                history.append({"role": "tool", "tool_call_id": call.id, "content": output})
                call = dataclasses.replace(call, output=output)
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

    def _stream(self, history: list[dict[str, Any]], config: ChatConfig) -> Iterator[dict[str, Any]]:
        messages = list(history)
        if config.instructions:
            messages.insert(0, {"role": "system", "content": config.instructions})
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "stream_options": {"include_usage": True},
            "temperature": config.temperature,
            "reasoning_effort": config.reasoning_effort,
        }
        if config.max_output_tokens:
            payload["max_tokens"] = config.max_output_tokens
        if config.tools:
            payload["tools"] = [t.to_openai() for t in config.tools]
        payload.update(self.extra_body)

        try:
            response = self.session.post(
                self.url, json=payload, headers=self.headers, stream=True, timeout=self.timeout
            )
        except requests.RequestException as e:
            raise GptOssError(f"Could not reach {self.url}: {e}") from e
        with response:
            if response.status_code >= 400:
                raise GptOssError(f"{self.url} returned {response.status_code}: {response.text[:2000]}")
            # Decode as UTF-8 ourselves: requests assumes ISO-8859-1 for text/* without a charset.
            for raw in response.iter_lines(chunk_size=None):
                line = raw.decode("utf-8").strip()
                if not line.startswith("data:"):
                    continue
                data = line[len("data:"):].strip()
                if data == "[DONE]":
                    return
                chunk = json.loads(data)
                if "error" in chunk:
                    raise GptOssError(f"{self.url} returned an error: {chunk['error']}")
                yield chunk


def _merge_tool_call_delta(pending: list[dict[str, str]], part: dict[str, Any]) -> None:
    """Accumulate a streamed tool call fragment into `pending`."""
    fn = part.get("function") or {}
    index = part.get("index")
    if index is None:
        # Some servers omit the index: a fragment with an id or name starts a new call.
        index = len(pending) if (part.get("id") or fn.get("name") or not pending) else len(pending) - 1
    while len(pending) <= index:
        pending.append({"id": "", "name": "", "arguments": ""})
    slot = pending[index]
    if part.get("id"):
        slot["id"] = part["id"]
    if fn.get("name"):
        slot["name"] += fn["name"]
    if fn.get("arguments"):
        slot["arguments"] += fn["arguments"] if isinstance(fn["arguments"], str) else json.dumps(fn["arguments"])
    if not slot["id"]:
        slot["id"] = f"call_{uuid.uuid4().hex}"
