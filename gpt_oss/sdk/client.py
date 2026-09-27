"""The public entry points: `GptOss` (a loaded model) and `Chat` (a conversation)."""

from __future__ import annotations

import asyncio
import datetime
import os
import platform
import threading
from collections.abc import AsyncIterator, Callable, Iterable, Iterator
from typing import Any

from .engine import ChatConfig, Engine, ReasoningEffort
from .harmony import HarmonyEngine, NextTokenGenerator, TokenGenerator
from .openai_compat import OpenAICompatibleEngine
from .tools import FunctionTool, as_function_tool, caller_loop
from .types import Event, GptOssError, Reply

LOCAL_BACKENDS = ("triton", "torch", "vllm", "metal", "transformers")


class GptOss:
    """A gpt-oss model, ready to be used from your own code.

    Pick how the model runs with one of the constructors::

        model = GptOss.ollama("gpt-oss:20b")                     # local Ollama server
        model = GptOss.openai_compatible("http://host:8000/v1", "openai/gpt-oss-120b")
        model = GptOss.local("gpt-oss-20b/original/", backend="triton")  # in-process

    then call `ask` for one-off questions or `chat` for a conversation.
    Keyword arguments given here become defaults for every chat.
    """

    def __init__(
        self,
        engine: Engine,
        *,
        reasoning_effort: ReasoningEffort = "medium",
        temperature: float = 1.0,
        max_output_tokens: int | None = None,
        max_tool_rounds: int = 8,
    ):
        self.engine = engine
        self.defaults = dict(
            reasoning_effort=reasoning_effort,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            max_tool_rounds=max_tool_rounds,
        )

    # -- constructors -------------------------------------------------------

    @classmethod
    def ollama(
        cls,
        model: str = "gpt-oss:20b",
        *,
        base_url: str = "http://localhost:11434/v1",
        **defaults: Any,
    ) -> GptOss:
        """Use a model served by a local Ollama (``ollama pull gpt-oss:20b``)."""
        return cls.openai_compatible(base_url, model, **defaults)

    @classmethod
    def openai_compatible(
        cls,
        base_url: str,
        model: str,
        *,
        api_key: str | None = None,
        timeout: float = 600.0,
        extra_body: dict[str, Any] | None = None,
        **defaults: Any,
    ) -> GptOss:
        """Use any OpenAI-compatible ``/chat/completions`` server (vLLM, LM Studio, ...)."""
        engine = OpenAICompatibleEngine(
            base_url, model, api_key=api_key, timeout=timeout, extra_body=extra_body
        )
        return cls(engine, **defaults)

    @classmethod
    def local(
        cls,
        checkpoint: str,
        backend: str = "auto",
        *,
        context: int = 8192,
        tensor_parallel_size: int = 1,
        **defaults: Any,
    ) -> GptOss:
        """Load the weights into this process.

        `backend` is one of ``triton``, ``torch``, ``vllm``, ``metal``,
        ``transformers`` or ``auto`` (metal on Apple Silicon, else triton).
        `context` is the KV-cache size used by the triton backend.
        """
        generator = load_generator(
            checkpoint, backend, context=context, tensor_parallel_size=tensor_parallel_size
        )
        return cls.from_generator(generator, **defaults)

    @classmethod
    def from_generator(cls, generator: TokenGenerator, **defaults: Any) -> GptOss:
        """Use any object with a ``generate(prompt_tokens, stop_tokens, temperature, max_tokens)``
        method yielding harmony token ids."""
        return cls(HarmonyEngine(generator), **defaults)

    @classmethod
    def from_next_token_fn(
        cls, infer_next_token: Callable[[list[int], float, bool], int], **defaults: Any
    ) -> GptOss:
        """Use an ``infer_next_token(tokens, temperature, new_request)`` function,
        like the ones in `gpt_oss.responses_api.inference`."""
        return cls.from_generator(NextTokenGenerator(infer_next_token), **defaults)

    # -- usage ----------------------------------------------------------------

    def chat(
        self,
        *,
        instructions: str | None = None,
        tools: Iterable[Any] = (),
        history: list[dict[str, Any]] | None = None,
        **overrides: Any,
    ) -> Chat:
        """Start a conversation.

        `tools` takes plain functions, `@tool`-decorated functions and, for
        local backends, harmony tools such as
        `gpt_oss.tools.simple_browser.SimpleBrowserTool`. `history` resumes a
        conversation saved with `Chat.history`. Other keyword arguments
        (``reasoning_effort``, ``temperature``, ``max_output_tokens``,
        ``max_tool_rounds``) override the model defaults.
        """
        function_tools, builtin_tools = _split_tools(tools)
        settings = {**self.defaults, **overrides}
        config = ChatConfig(
            instructions=instructions,
            tools=function_tools,
            builtin_tools=builtin_tools,
            **settings,
        )
        # Fixed per chat so the prompt prefix (and any KV cache) stays stable.
        config.start_date = config.start_date or datetime.date.today().isoformat()
        if config.reasoning_effort not in ("low", "medium", "high"):
            raise ValueError(f"reasoning_effort must be low, medium or high, not {config.reasoning_effort!r}")
        return Chat(self.engine, config, history)

    def ask(self, prompt: str, **chat_kwargs: Any) -> Reply:
        """One-off question: start a fresh chat, send `prompt`, return the reply."""
        return self.chat(**chat_kwargs).send(prompt)

    async def aask(self, prompt: str, **chat_kwargs: Any) -> Reply:
        return await self.chat(**chat_kwargs).asend(prompt)


class Chat:
    """A conversation. Keeps the history between `send` calls.

    A chat handles one message at a time; use one chat per user/thread.
    """

    def __init__(self, engine: Engine, config: ChatConfig, history: list[dict[str, Any]] | None = None):
        self.engine = engine
        self.config = config
        self._history = engine.load_history(history) if history else []
        self._busy = threading.Lock()

    @property
    def history(self) -> list[dict[str, Any]]:
        """The conversation as JSON-serializable dicts; pass it back to
        `GptOss.chat(history=...)` to resume it later."""
        return self.engine.dump_history(self._history)

    def reset(self) -> None:
        self._history.clear()

    def send(self, message: str) -> Reply:
        reply = None
        for event in self.stream(message):
            if event.type == "done":
                reply = event.reply
        if reply is None:
            raise GptOssError("The backend finished without a reply")
        return reply

    def stream(self, message: str) -> Iterator[Event]:
        """Send `message` and yield events as the model produces them.

        If the caller stops iterating before the ``done`` event, the turn is
        discarded from the history.
        """
        if not self._busy.acquire(blocking=False):
            raise RuntimeError("This chat is already handling a message")
        checkpoint = len(self._history)
        completed = False
        try:
            for event in self.engine.run(self._history, message, self.config):
                completed = event.type == "done"
                yield event
        finally:
            if not completed:
                del self._history[checkpoint:]
            self._busy.release()

    async def asend(self, message: str) -> Reply:
        reply = None
        async for event in self.astream(message):
            if event.type == "done":
                reply = event.reply
        if reply is None:
            raise GptOssError("The backend finished without a reply")
        return reply

    async def astream(self, message: str) -> AsyncIterator[Event]:
        """Async version of `stream`. The model runs on a worker thread, and
        async tools run on the calling event loop."""
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[Any] = asyncio.Queue()
        finished = loop.create_future()
        stop = threading.Event()
        end = object()

        def put(item: Any) -> None:
            try:
                loop.call_soon_threadsafe(queue.put_nowait, item)
            except RuntimeError:  # the loop is closed; nobody is listening
                pass

        def worker() -> None:
            events = self.stream(message)
            try:
                with caller_loop(loop):
                    for event in events:
                        put(event)
                        if stop.is_set():
                            break
            except BaseException as e:
                put(_Failure(e))
            finally:
                events.close()
                put(end)
                try:
                    loop.call_soon_threadsafe(lambda: finished.done() or finished.set_result(None))
                except RuntimeError:
                    pass

        threading.Thread(target=worker, name="gpt-oss-chat", daemon=True).start()
        try:
            while (item := await queue.get()) is not end:
                if isinstance(item, _Failure):
                    raise item.error
                yield item
        finally:
            stop.set()
            await finished


class _Failure:
    def __init__(self, error: BaseException):
        self.error = error


def _split_tools(tools: Iterable[Any]) -> tuple[list[FunctionTool], list[Any]]:
    from gpt_oss.tools.tool import Tool

    function_tools: list[FunctionTool] = []
    builtin_tools: list[Any] = []
    for t in tools:
        if isinstance(t, Tool):
            builtin_tools.append(t)
        else:
            function_tools.append(as_function_tool(t))
    names = [t.name for t in function_tools] + [t.name for t in builtin_tools]
    duplicates = {n for n in names if names.count(n) > 1}
    if duplicates:
        raise ValueError(f"Duplicate tool names: {sorted(duplicates)}")
    return function_tools, builtin_tools


def load_generator(
    checkpoint: str, backend: str = "auto", *, context: int = 8192, tensor_parallel_size: int = 1
) -> TokenGenerator:
    """Load one of this repository's inference implementations as a `TokenGenerator`."""
    checkpoint = os.path.expanduser(checkpoint)
    if backend == "auto":
        on_apple_silicon = platform.system() == "Darwin" and platform.machine() == "arm64"
        backend = "metal" if on_apple_silicon else "triton"
    if backend in ("triton", "torch"):
        device = _torch_device()
        if backend == "triton":
            from gpt_oss.triton.model import TokenGenerator as TritonGenerator

            return TritonGenerator(checkpoint, context=context, device=device)
        from gpt_oss.torch.model import TokenGenerator as TorchGenerator

        return TorchGenerator(checkpoint, device=device)
    if backend == "vllm":
        from gpt_oss.vllm.token_generator import TokenGenerator as VLLMGenerator

        return VLLMGenerator(checkpoint, tensor_parallel_size=tensor_parallel_size)
    if backend == "metal":
        from gpt_oss.responses_api.inference.metal import setup_model
    elif backend == "transformers":
        from gpt_oss.responses_api.inference.transformers import setup_model
    else:
        raise ValueError(f"Unknown backend {backend!r}; expected 'auto' or one of {LOCAL_BACKENDS}")
    return NextTokenGenerator(setup_model(checkpoint))


def _torch_device() -> Any:
    if int(os.environ.get("WORLD_SIZE", 1)) > 1:
        from gpt_oss.torch.utils import init_distributed

        return init_distributed()
    import torch

    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0)))
    torch.cuda.set_device(device)
    return device
