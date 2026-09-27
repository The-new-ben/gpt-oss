# The SDK is imported lazily so `import gpt_oss.torch` etc. stay lightweight.
_SDK_EXPORTS = {"Chat", "Event", "GptOss", "GptOssError", "Reply", "ToolCall", "tool"}


def __getattr__(name):
    if name in _SDK_EXPORTS:
        from . import sdk

        return getattr(sdk, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
