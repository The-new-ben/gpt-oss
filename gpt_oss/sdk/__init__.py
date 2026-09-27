"""Use gpt-oss as a library inside your own project.

    from gpt_oss import GptOss, tool

    @tool
    def get_weather(city: str) -> str:
        '''Get the current weather for a city.'''
        return "sunny, 21C"

    model = GptOss.ollama("gpt-oss:20b")
    print(model.ask("What's the weather in Paris?", tools=[get_weather]))
"""

from .client import Chat, GptOss, load_generator
from .engine import ChatConfig, Engine
from .harmony import HarmonyEngine, NextTokenGenerator, TokenGenerator
from .openai_compat import OpenAICompatibleEngine
from .tools import FunctionTool, tool
from .types import Event, GptOssError, Reply, ToolCall, Usage

__all__ = [
    "Chat",
    "ChatConfig",
    "Engine",
    "Event",
    "FunctionTool",
    "GptOss",
    "GptOssError",
    "HarmonyEngine",
    "NextTokenGenerator",
    "OpenAICompatibleEngine",
    "Reply",
    "TokenGenerator",
    "ToolCall",
    "Usage",
    "load_generator",
    "tool",
]
