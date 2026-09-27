"""Embed gpt-oss in a Python program with tools, streaming and saved history.

Start a model server first, for example:

    ollama pull gpt-oss:20b

Then run:

    python examples/sdk/quickstart.py
    python examples/sdk/quickstart.py --base-url http://localhost:8000/v1 --model openai/gpt-oss-20b  # vllm serve
    python examples/sdk/quickstart.py --checkpoint gpt-oss-20b/original/ --backend triton             # in-process
"""

import argparse
import asyncio
import datetime
import json
from typing import Literal

from gpt_oss import GptOss, tool


@tool
def get_weather(city: str, unit: Literal["celsius", "fahrenheit"] = "celsius") -> dict:
    """Get the current weather for a city.

    Args:
        city: City name, e.g. "Paris".
        unit: Temperature unit.
    """
    # Replace with a real weather API call.
    temperature = 21 if unit == "celsius" else 70
    return {"city": city, "temperature": temperature, "unit": unit, "sky": "sunny"}


@tool
async def current_time(timezone_offset_hours: int = 0) -> str:
    """Get the current time at a UTC offset."""
    await asyncio.sleep(0)  # async tools work too
    tz = datetime.timezone(datetime.timedelta(hours=timezone_offset_hours))
    return datetime.datetime.now(tz).isoformat(timespec="minutes")


def load_model(args: argparse.Namespace) -> GptOss:
    if args.checkpoint:
        return GptOss.local(args.checkpoint, backend=args.backend)
    return GptOss.openai_compatible(args.base_url, args.model)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", default="http://localhost:11434/v1", help="OpenAI-compatible server URL")
    parser.add_argument("--model", default="gpt-oss:20b", help="Model name on the server")
    parser.add_argument("--checkpoint", help="Run in-process from this checkpoint instead of a server")
    parser.add_argument("--backend", default="auto", help="In-process backend (triton, torch, vllm, metal, transformers)")
    args = parser.parse_args()

    model = load_model(args)

    # 1. One-off question.
    reply = model.ask("What's the weather in Paris?", tools=[get_weather], reasoning_effort="low")
    print("Answer:", reply.text)
    for call in reply.tool_calls:
        print(f"  called {call.name}({call.arguments}) -> {call.output}")

    # 2. A streaming conversation.
    chat = model.chat(
        instructions="You are a concise travel assistant.",
        tools=[get_weather, current_time],
    )
    for message in ["What time is it in Tokyo (UTC+9)?", "And is it warm there?"]:
        print(f"\nUser: {message}\nAssistant: ", end="")
        for event in chat.stream(message):
            if event.type == "text":
                print(event.text, end="", flush=True)
            elif event.type == "tool_call":
                print(f"[{event.tool_call.name}] ", end="", flush=True)
        print()

    # 3. Save the conversation and pick it up again later.
    saved = json.dumps(chat.history)
    resumed = model.chat(tools=[get_weather, current_time], history=json.loads(saved))
    print("\nResumed:", resumed.send("Summarize our conversation in one sentence.").text)


if __name__ == "__main__":
    main()
