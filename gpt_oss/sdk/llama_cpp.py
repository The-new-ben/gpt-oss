"""Run GGUF checkpoints (e.g. ggml-org/gpt-oss-20b-GGUF) through llama.cpp.

Works on CPUs, Apple Silicon and consumer GPUs without a separate server:

    pip install llama-cpp-python
    model = GptOss.local("gpt-oss-20b-MXFP4.gguf", backend="llama_cpp")
"""

from __future__ import annotations

import ctypes
import threading
from collections.abc import Iterator
from typing import Any


class LlamaCppGenerator:
    """`TokenGenerator` backed by `llama_cpp.Llama`.

    llama.cpp reuses the KV cache for the longest common prompt prefix, so
    multi-turn chats only pay for the new tokens of each turn. For several
    agents sharing one model, give each its own `fork()`: forks share the
    weights (no extra memory) but keep separate prompt caches, which are
    swapped in and out of llama.cpp as the agents take turns.
    """

    def __init__(self, model_path: str | None = None, *, context: int = 8192, llm: Any = None, **llama_kwargs: Any):
        if llm is None:
            from llama_cpp import Llama

            llama_kwargs.setdefault("verbose", False)
            llm = Llama(model_path=model_path, n_ctx=context, **llama_kwargs)
            llm._gpt_oss_lock = threading.Lock()
            llm._gpt_oss_owner = None
        self.llm = llm
        self._saved: tuple[bytes, Any, int] | None = None  # our KV cache while another fork runs

    def fork(self) -> LlamaCppGenerator:
        """Another generator on the same weights with its own prompt cache."""
        return LlamaCppGenerator(llm=self.llm)

    def generate(
        self,
        prompt_tokens: list[int],
        stop_tokens: list[int],
        temperature: float = 1.0,
        max_tokens: int = 0,
    ) -> Iterator[int]:
        with self.llm._gpt_oss_lock:
            self._activate()
            # gpt-oss is meant to be sampled with plain temperature (top_p=1, no top_k/min_p).
            tokens = self.llm.generate(
                prompt_tokens, temp=temperature, top_k=0, top_p=1.0, min_p=0.0, repeat_penalty=1.0
            )
            for count, token in enumerate(tokens, start=1):
                yield token
                if token in stop_tokens or (max_tokens and count >= max_tokens):
                    return

    def _activate(self) -> None:
        owner = self.llm._gpt_oss_owner
        if owner is self:
            return
        if owner is not None:
            owner._saved = _save_kv(self.llm)
        if self._saved is not None:
            _load_kv(self.llm, self._saved)
        else:
            self.llm.reset()
        self.llm._gpt_oss_owner = self


def _save_kv(llm: Any) -> tuple[bytes, Any, int]:
    # Only the llama.cpp context state and the token ids: Llama.save_state() would
    # also copy the logits buffer (hundreds of MB), which sampling doesn't need.
    import llama_cpp

    size = llama_cpp.llama_state_get_size(llm._ctx.ctx)
    buffer = (ctypes.c_uint8 * size)()
    written = llama_cpp.llama_state_get_data(llm._ctx.ctx, buffer, size)
    return bytes(memoryview(buffer)[:written]), llm.input_ids[: llm.n_tokens].copy(), llm.n_tokens


def _load_kv(llm: Any, saved: tuple[bytes, Any, int]) -> None:
    import llama_cpp

    data, token_ids, n_tokens = saved
    buffer = (ctypes.c_uint8 * len(data)).from_buffer_copy(data)
    llama_cpp.llama_state_set_data(llm._ctx.ctx, buffer, len(data))
    llm.input_ids[:n_tokens] = token_ids
    llm.n_tokens = n_tokens
