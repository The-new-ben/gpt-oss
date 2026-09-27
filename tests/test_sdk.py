import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Annotated, Literal

import pydantic
import pytest
from openai_harmony import Author, Message, Role, TextContent, ToolNamespaceConfig

import gpt_oss
from gpt_oss.sdk import GptOss, GptOssError, NextTokenGenerator, tool
from gpt_oss.tools.tool import Tool


FINAL_ONLY = "<|channel|>analysis<|message|>User greets.<|end|><|start|>assistant<|channel|>final<|message|>Hello there!<|return|>"
WEATHER_CALL = (
    "<|channel|>analysis<|message|>Need weather.<|end|>"
    '<|start|>assistant<|channel|>commentary to=functions.get_weather <|constrain|>json<|message|>{"city":"Paris"}<|call|>'
)
WEATHER_ANSWER = "<|channel|>final<|message|>It is sunny in Paris.<|return|>"


class ScriptedGenerator:
    """Stands in for a local backend: replays one scripted completion per call."""

    def __init__(self, encoding, completions):
        self.encoding = encoding
        self.completions = list(completions)
        self.prompts = []

    def generate(self, prompt_tokens, stop_tokens, temperature=1.0, max_tokens=0):
        self.prompts.append(self.encoding.decode(list(prompt_tokens)))
        tokens = self.encoding.encode(self.completions.pop(0), allowed_special="all")
        for i, token in enumerate(tokens):
            if max_tokens and i >= max_tokens:
                return
            yield token
            if token in stop_tokens:
                return


@pytest.fixture
def local_model(harmony_encoding):
    def make(*completions, **defaults):
        generator = ScriptedGenerator(harmony_encoding, completions)
        return GptOss.from_generator(generator, **defaults), generator

    return make


def weather_tool(calls):
    @tool
    def get_weather(city: str) -> dict:
        """Get the current weather.

        Args:
            city: City name.
        """
        calls.append(city)
        return {"city": city, "sky": "sunny"}

    return get_weather


# -- tool schemas -------------------------------------------------------------


class Point(pydantic.BaseModel):
    x: int
    y: int


def test_tool_schema_from_type_hints_and_docstring():
    @tool
    def plot(
        points: list[Point],
        style: Literal["line", "bar"] = "line",
        title: Annotated[str, "Chart title"] = "",
        scale: float | None = None,
    ) -> str:
        """Draw a chart.

        Args:
            points: The points to draw,
                in order.
            style: Chart style.

        Returns:
            A URL.
        """
        return "ok"

    assert plot.name == "plot"
    assert plot.description == "Draw a chart."
    params = plot.parameters
    assert params["required"] == ["points"]
    assert params["properties"]["points"]["type"] == "array"
    assert params["properties"]["points"]["items"]["properties"] == {"x": {"type": "integer"}, "y": {"type": "integer"}}
    assert params["properties"]["points"]["description"] == "The points to draw, in order."
    assert params["properties"]["style"] == {
        "enum": ["line", "bar"],
        "type": "string",
        "description": "Chart style.",
        "default": "line",
    }
    assert params["properties"]["title"]["description"] == "Chart title"
    assert params["properties"]["scale"] == {"type": "number"}
    assert plot([]) == "ok"  # still callable as a normal function


class Route(pydantic.BaseModel):
    start: Point
    end: Point


class Tree(pydantic.BaseModel):
    label: str
    children: list["Tree"] = []


def test_nested_models_are_inlined():
    @tool
    def plan(routes: list[Route], tree: Tree | None = None) -> str:
        return "ok"

    params = plan.parameters
    start = params["properties"]["routes"]["items"]["properties"]["start"]
    assert start == {"properties": {"x": {"type": "integer"}, "y": {"type": "integer"}}, "required": ["x", "y"], "type": "object"}
    # Recursive references can't be inlined, so they are left untyped.
    assert params["properties"]["tree"]["properties"]["children"]["items"] == {}
    assert "$ref" not in json.dumps(params) and "$defs" not in json.dumps(params)
    assert plan.invoke({"routes": [{"start": {"x": 0, "y": 0}, "end": {"x": 1, "y": 1}}]}) == "ok"


def test_tool_invoke_handles_models_async_and_errors():
    @tool(name="area", description="Area of a rectangle")
    def area(corner: Point, other: Point) -> int:
        return abs(corner.x - other.x) * abs(corner.y - other.y)

    assert area.invoke('{"corner": {"x": 0, "y": 0}, "other": {"x": 2, "y": 3}}') == "6"
    assert area.invoke("not json").startswith("Error: JSONDecodeError")
    assert area.invoke('{"corner": {"x": 0}}').startswith("Error: ValidationError")

    @tool
    async def echo(text: str) -> Point:
        await asyncio.sleep(0)
        return Point(x=len(text), y=0)

    assert json.loads(echo.invoke({"text": "abc"})) == {"x": 3, "y": 0}


def test_invalid_tool_name():
    with pytest.raises(ValueError):
        tool(name="has space")(lambda: None)


def test_top_level_exports():
    assert gpt_oss.GptOss is GptOss
    assert gpt_oss.tool is tool


# -- local (harmony) engine -----------------------------------------------------


def test_ask_streams_reasoning_then_text(local_model):
    model, generator = local_model(FINAL_ONLY, reasoning_effort="high")
    chat = model.chat(instructions="Be brief.")
    events = list(chat.stream("hi"))

    assert "".join(e.text for e in events if e.type == "reasoning") == "User greets."
    assert "".join(e.text for e in events if e.type == "text") == "Hello there!"
    assert events[-1].type == "done"
    reply = events[-1].reply
    assert str(reply) == "Hello there!"
    assert reply.reasoning == "User greets."
    assert reply.finish_reason == "stop"
    assert reply.usage.completion_tokens > 0
    assert "Reasoning: high" in generator.prompts[0]
    assert "Be brief." in generator.prompts[0]


def test_function_tool_loop(local_model):
    calls = []
    model, generator = local_model(WEATHER_CALL, WEATHER_ANSWER)
    reply = model.ask("Weather in Paris?", tools=[weather_tool(calls)])

    assert calls == ["Paris"]
    assert reply.text == "It is sunny in Paris."
    assert [c.name for c in reply.tool_calls] == ["get_weather"]
    assert reply.tool_calls[0].args == {"city": "Paris"}
    assert json.loads(reply.tool_calls[0].output) == {"city": "Paris", "sky": "sunny"}
    assert "namespace functions" in generator.prompts[0]
    assert "type get_weather" in generator.prompts[0]
    # The tool result is fed back to the model in the second round.
    assert "<|start|>functions.get_weather to=assistant<|channel|>commentary" in generator.prompts[1]
    assert '"sky": "sunny"' in generator.prompts[1]


def test_plain_functions_are_accepted_as_tools(local_model):
    def get_weather(city: str) -> str:
        return f"rain in {city}"

    model, generator = local_model(WEATHER_CALL, WEATHER_ANSWER)
    reply = model.ask("Weather?", tools=[get_weather])
    assert reply.tool_calls[0].output == "rain in Paris"


def test_unknown_function_is_reported_to_the_model(local_model):
    model, generator = local_model(WEATHER_CALL, WEATHER_ANSWER)
    reply = model.ask("Weather?")
    assert reply.tool_calls[0].output == "Error: unknown function 'get_weather'"
    assert "unknown function" in generator.prompts[1]


def test_max_tool_rounds(local_model):
    calls = []
    model, generator = local_model(WEATHER_CALL, WEATHER_CALL, WEATHER_CALL, max_tool_rounds=2)
    chat = model.chat(tools=[weather_tool(calls)])
    reply = chat.send("Weather?")
    assert reply.finish_reason == "max_tool_rounds"
    assert calls == ["Paris", "Paris"]
    # The unexecuted call still gets a result so the history stays well formed.
    assert chat.history[-1]["content"][0]["text"] == "Error: tool call limit reached"


def test_max_output_tokens_truncates(local_model):
    model, generator = local_model(FINAL_ONLY, max_output_tokens=12)
    chat = model.chat()
    reply = chat.send("hi")
    assert reply.finish_reason == "length"
    assert reply.text == ""
    assert reply.reasoning.startswith("User")


def test_history_persists_and_resumes(local_model, harmony_encoding):
    model, generator = local_model(FINAL_ONLY, WEATHER_ANSWER)
    chat = model.chat()
    chat.send("hi")
    saved = json.loads(json.dumps(chat.history))
    assert [m["role"] for m in saved] == ["user", "assistant", "assistant"]

    resumed = model.chat(history=saved)
    resumed.send("and the weather?")
    assert "<|start|>user<|message|>hi<|end|>" in generator.prompts[1]
    assert "Hello there!" in generator.prompts[1]
    # Reasoning from finished turns is dropped, as the harmony format expects.
    assert "User greets." not in generator.prompts[1]


def test_abandoned_stream_rolls_back(local_model):
    model, generator = local_model(FINAL_ONLY, FINAL_ONLY)
    chat = model.chat()
    for event in chat.stream("hi"):
        break
    assert chat.history == []
    assert chat.send("hi again").text == "Hello there!"
    assert len(chat.history) == 3


class FakeBrowser(Tool):
    def __init__(self):
        self.seen = []

    @property
    def name(self):
        return "browser"

    @property
    def tool_config(self):
        return ToolNamespaceConfig.browser()

    def instruction(self):
        return "browse"

    async def _process(self, message):
        self.seen.append(message.content[0].text)
        yield Message(
            author=Author(role=Role.TOOL, name="browser.search"),
            content=[TextContent(text="[0] Result page")],
        ).with_recipient("assistant")


def test_builtin_harmony_tool(local_model):
    browser = FakeBrowser()
    search = (
        "<|channel|>analysis to=browser.search <|constrain|>json<|message|>"
        '{"query":"gpt-oss"}<|call|>'
    )
    model, generator = local_model(search, WEATHER_ANSWER)
    reply = model.ask("Search", tools=[browser])
    assert browser.seen == ['{"query":"gpt-oss"}']
    assert reply.tool_calls[0].name == "browser.search"
    assert reply.tool_calls[0].output == "[0] Result page"
    assert "namespace browser" in generator.prompts[0]
    assert "[0] Result page" in generator.prompts[1]


def test_async_api_runs_async_tools_on_caller_loop(local_model):
    seen_loops = []

    @tool
    async def get_weather(city: str) -> str:
        seen_loops.append(asyncio.get_running_loop())
        return "sunny"

    model, generator = local_model(WEATHER_CALL, WEATHER_ANSWER)

    async def main():
        chat = model.chat(tools=[get_weather])
        types = [e.type async for e in chat.astream("Weather?")]
        return asyncio.get_running_loop(), types

    loop, types = asyncio.run(main())
    assert seen_loops == [loop]
    assert [t for t in types if t != "reasoning" and t != "text"] == ["tool_call", "tool_result", "done"]
    assert types.index("tool_result") < types.index("text")


def test_sync_api_runs_async_tools(local_model):
    @tool
    async def get_weather(city: str) -> str:
        return "sunny"

    model, generator = local_model(WEATHER_CALL, WEATHER_ANSWER)
    assert model.ask("Weather?", tools=[get_weather]).tool_calls[0].output == "sunny"


def test_async_errors_propagate(local_model):
    model, generator = local_model()  # no scripted completions -> IndexError

    async def main():
        await model.aask("hi")

    with pytest.raises(IndexError):
        asyncio.run(main())


def test_chat_rejects_concurrent_messages(local_model):
    model, generator = local_model(FINAL_ONLY)
    chat = model.chat()
    stream = chat.stream("hi")
    next(stream)
    with pytest.raises(RuntimeError):
        chat.send("again")
    stream.close()


def test_duplicate_tool_names(local_model):
    model, generator = local_model()
    with pytest.raises(ValueError):
        model.chat(tools=[weather_tool([]), weather_tool([])])


def test_next_token_adapter(harmony_encoding):
    stop = harmony_encoding.stop_tokens_for_assistant_actions()
    script = harmony_encoding.encode("<|channel|>final<|message|>ok<|return|>", allowed_special="all")
    requests = []

    def infer_next_token(tokens, temperature, new_request):
        requests.append(new_request)
        return script[len(requests) - 1]

    tokens = list(NextTokenGenerator(infer_next_token).generate([1, 2], stop_tokens=stop))
    assert tokens == script
    assert requests == [True] + [False] * (len(script) - 1)
    assert GptOss.from_next_token_fn(infer_next_token) is not None


# -- OpenAI-compatible engine ---------------------------------------------------


def sse_chunks(*deltas, finish_reason="stop", usage=None):
    chunks = [{"choices": [{"index": 0, "delta": d, "finish_reason": None}]} for d in deltas]
    chunks.append({"choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}]})
    if usage:
        chunks.append({"choices": [], "usage": usage})
    return chunks


@pytest.fixture
def chat_server():
    """A tiny OpenAI-compatible server that replays scripted SSE responses."""
    script = []
    received = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            received.append({"path": self.path, "headers": dict(self.headers), "body": body})
            status, chunks = script.pop(0)
            self.send_response(status)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            if status >= 400:
                self.wfile.write(b'{"error": "bad model"}')
                return
            for chunk in chunks:
                self.wfile.write(f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n".encode())
            self.wfile.write(b"data: [DONE]\n\n")

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}/v1", script, received
    server.shutdown()


def test_openai_compatible_chat(chat_server):
    url, script, received = chat_server
    script.append((200, sse_chunks(
        {"role": "assistant", "reasoning_content": "Say hi."},
        {"content": "Bonjour, "},
        {"content": "café ☀"},
        usage={"prompt_tokens": 11, "completion_tokens": 5},
    )))
    model = GptOss.openai_compatible(url, "gpt-oss-20b", api_key="sk-test", reasoning_effort="low")
    chat = model.chat(instructions="Answer in French.")
    events = list(chat.stream("hi"))

    reply = events[-1].reply
    assert reply.text == "Bonjour, café ☀"
    assert reply.reasoning == "Say hi."
    assert reply.usage.prompt_tokens == 11
    assert [e.type for e in events] == ["reasoning", "text", "text", "done"]

    request = received[0]
    assert request["path"] == "/v1/chat/completions"
    assert request["headers"]["Authorization"] == "Bearer sk-test"
    assert request["body"]["model"] == "gpt-oss-20b"
    assert request["body"]["reasoning_effort"] == "low"
    assert request["body"]["stream_options"] == {"include_usage": True}
    assert request["body"]["messages"] == [
        {"role": "system", "content": "Answer in French."},
        {"role": "user", "content": "hi"},
    ]
    assert chat.history[-1] == {"role": "assistant", "content": "Bonjour, café ☀"}


def test_openai_compatible_tool_loop(chat_server):
    url, script, received = chat_server
    script.append((200, sse_chunks(
        {"reasoning": "Need weather."},
        {"tool_calls": [{"index": 0, "id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": '{"ci'}}]},
        {"tool_calls": [{"index": 0, "function": {"arguments": 'ty": "Paris"}'}}]},
        finish_reason="tool_calls",
    )))
    script.append((200, sse_chunks({"content": "Sunny."})))
    calls = []
    model = GptOss.ollama(base_url=url)
    reply = model.ask("Weather?", tools=[weather_tool(calls)])

    assert calls == ["Paris"]
    assert reply.text == "Sunny."
    assert reply.tool_calls[0].id == "call_1"
    first, second = received[0]["body"], received[1]["body"]
    assert first["model"] == "gpt-oss:20b"
    assert first["tools"][0]["function"]["name"] == "get_weather"
    assert first["tools"][0]["function"]["parameters"]["required"] == ["city"]
    assert second["messages"][-2]["tool_calls"][0]["function"] == {
        "name": "get_weather",
        "arguments": '{"city": "Paris"}',
    }
    assert second["messages"][-1]["role"] == "tool"
    assert second["messages"][-1]["tool_call_id"] == "call_1"
    assert json.loads(second["messages"][-1]["content"])["sky"] == "sunny"


def test_openai_compatible_http_error(chat_server):
    url, script, received = chat_server
    script.append((404, []))
    chat = GptOss.openai_compatible(url, "missing").chat()
    with pytest.raises(GptOssError, match="404"):
        chat.send("hi")
    assert chat.history == []


def test_openai_compatible_rejects_builtin_tools(chat_server):
    url, script, received = chat_server
    with pytest.raises(GptOssError, match="local backend"):
        GptOss.openai_compatible(url, "m").ask("hi", tools=[FakeBrowser()])


def test_ends_turn_tool_stops_the_turn(local_model, harmony_encoding):
    captured = []

    @tool(ends_turn=True)
    def submit(answer: str) -> str:
        captured.append(answer)
        return "saved"

    submit_call = '<|channel|>commentary to=functions.submit <|constrain|>json<|message|>{"answer":"42"}<|call|>'
    model, generator = local_model(submit_call)  # a second generation would fail: nothing scripted
    reply = model.ask("Answer", tools=[submit])
    assert captured == ["42"]
    assert reply.finish_reason == "tool"
    assert len(generator.prompts) == 1


def test_ends_turn_over_http(chat_server):
    url, script, received = chat_server
    script.append((200, sse_chunks(
        {"tool_calls": [{"index": 0, "id": "c1", "function": {"name": "submit", "arguments": '{"answer": "42"}'}}]},
        finish_reason="tool_calls",
    )))

    @tool(ends_turn=True)
    def submit(answer: str) -> str:
        return "saved"

    reply = GptOss.openai_compatible(url, "m").ask("Answer", tools=[submit])
    assert reply.finish_reason == "tool"
    assert reply.tool_calls[0].output == "saved"
    assert len(received) == 1


def test_llama_cpp_backend_is_selected_for_gguf(monkeypatch):
    from gpt_oss.sdk import client, llama_cpp

    created = {}

    class FakeLlama(llama_cpp.LlamaCppGenerator):
        def __init__(self, path, *, context):
            created.update(path=path, context=context)

    monkeypatch.setattr(llama_cpp, "LlamaCppGenerator", FakeLlama)
    generator = client.load_generator("model.gguf", context=2048)
    assert isinstance(generator, FakeLlama)
    assert created == {"path": "model.gguf", "context": 2048}


def test_llama_cpp_forks_swap_prompt_caches(monkeypatch):
    import threading

    from gpt_oss.sdk import llama_cpp

    class FakeLlama:
        def __init__(self):
            self._gpt_oss_lock = threading.Lock()
            self._gpt_oss_owner = None
            self.cache = None  # stands in for the KV cache contents
            self.resets = 0

        def reset(self):
            self.cache = None
            self.resets += 1

        def generate(self, tokens, **kwargs):
            self.cache = list(tokens)
            yield 7
            yield 99

    monkeypatch.setattr(llama_cpp, "_save_kv", lambda llm: llm.cache)
    monkeypatch.setattr(llama_cpp, "_load_kv", lambda llm, saved: setattr(llm, "cache", saved))

    llm = FakeLlama()
    a = llama_cpp.LlamaCppGenerator(llm=llm)
    b = a.fork()
    assert list(a.generate([1, 2], stop_tokens=[99])) == [7, 99]
    assert list(b.generate([3, 4], stop_tokens=[99])) == [7, 99]
    assert a._saved == [1, 2]  # a's cache was saved when b took over
    list(a.generate([1, 2, 5], stop_tokens=[99]))
    assert b._saved == [3, 4] and llm.cache == [1, 2, 5]
    assert llm.resets == 2  # a and b each started from an empty cache once
    assert list(b.generate([3], stop_tokens=[99], max_tokens=1)) == [7]
