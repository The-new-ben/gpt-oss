"""Turn ordinary Python functions into tools the model can call."""

from __future__ import annotations

import asyncio
import dataclasses
import enum
import inspect
import json
import re
import threading
import types
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from typing import Annotated, Any, Literal, Union, get_args, get_origin, get_type_hints

import pydantic

_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_PRIMITIVES = {str: "string", int: "integer", float: "number", bool: "boolean"}
_ARGS_HEADER = re.compile(r"^(Args|Arguments|Parameters|Params):$")
_SECTION_HEADER = re.compile(r"^[A-Z][A-Za-z ]*:$")
_ARG_LINE = re.compile(r"^(\w+)\s*(?:\([^)]*\))?\s*:\s*(.*)$")


@dataclasses.dataclass
class FunctionTool:
    """A Python callable described with a JSON schema.

    Create one with the `tool` decorator. Calling the object still calls the
    wrapped function, so decorated functions keep working as normal code.
    """

    name: str
    description: str
    parameters: dict[str, Any]
    fn: Callable[..., Any]
    # Parameters annotated with a pydantic model are validated into that model.
    _models: dict[str, type[pydantic.BaseModel]] = dataclasses.field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if not _NAME_PATTERN.match(self.name):
            raise ValueError(
                f"Invalid tool name {self.name!r}: use 1-64 letters, digits, '_' or '-'"
            )

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.fn(*args, **kwargs)

    def to_openai(self) -> dict[str, Any]:
        """The tool in OpenAI chat-completions format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def invoke(self, arguments: str | Mapping[str, Any]) -> str:
        """Run the tool with model-produced arguments and return its output as text.

        Errors are returned as text instead of raised, so the model can see
        what went wrong and try again.
        """
        try:
            if isinstance(arguments, str):
                kwargs = json.loads(arguments) if arguments.strip() else {}
            else:
                kwargs = dict(arguments)
            if not isinstance(kwargs, dict):
                raise ValueError("arguments must be a JSON object")
            for key, model in self._models.items():
                if key in kwargs:
                    kwargs[key] = model.model_validate(kwargs[key])
            result = self.fn(**kwargs)
            if inspect.isawaitable(result):
                result = run_awaitable(result)
        except Exception as e:
            return f"Error: {type(e).__name__}: {e}"
        return _to_text(result)


def tool(
    fn: Callable[..., Any] | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
) -> Any:
    """Decorator that exposes a function to the model.

    The JSON schema is built from the type hints, and the description (plus
    per-argument descriptions under an ``Args:`` section) from the docstring::

        @tool
        def get_weather(city: str, unit: Literal["c", "f"] = "c") -> str:
            '''Get the current weather.

            Args:
                city: City name, e.g. "Paris".
            '''

    Sync and async functions are both supported.
    """

    def wrap(f: Callable[..., Any]) -> FunctionTool:
        doc_description, arg_docs = _parse_docstring(f.__doc__)
        parameters, models = _parameters_schema(f, arg_docs)
        return FunctionTool(
            name=name or f.__name__,
            description=description if description is not None else doc_description,
            parameters=parameters,
            fn=f,
            _models=models,
        )

    return wrap(fn) if fn is not None else wrap


def as_function_tool(obj: FunctionTool | Callable[..., Any]) -> FunctionTool:
    if isinstance(obj, FunctionTool):
        return obj
    if callable(obj):
        return tool(obj)
    raise TypeError(f"Not a tool: {obj!r}")


# --------------------------------------------------------------------------
# Running async tools from sync code
# --------------------------------------------------------------------------

_local = threading.local()


@contextmanager
def caller_loop(loop: asyncio.AbstractEventLoop):
    """Run async tools on `loop` while inside this block (used by the async API)."""
    previous = getattr(_local, "loop", None)
    _local.loop = loop
    try:
        yield
    finally:
        _local.loop = previous


def run_awaitable(awaitable: Any) -> Any:
    async def _await() -> Any:
        return await awaitable

    loop = getattr(_local, "loop", None)
    if loop is not None and loop.is_running():
        # Called from a worker thread of the async API: run on the caller's loop so
        # the tool can use resources (sessions, clients) bound to it.
        return asyncio.run_coroutine_threadsafe(_await(), loop).result()
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(_await())
    # Sync API called from inside a running loop: use a private loop in a helper thread.
    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, _await()).result()


# --------------------------------------------------------------------------
# Schema generation
# --------------------------------------------------------------------------


def _to_text(result: Any) -> str:
    if isinstance(result, str):
        return result
    return json.dumps(result, default=_json_default, ensure_ascii=False)


def _json_default(value: Any) -> Any:
    if isinstance(value, pydantic.BaseModel):
        return value.model_dump(mode="json")
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return dataclasses.asdict(value)
    if isinstance(value, enum.Enum):
        return value.value
    return str(value)


def _parse_docstring(doc: str | None) -> tuple[str, dict[str, str]]:
    lines = inspect.cleandoc(doc or "").splitlines()
    description: list[str] = []
    args: dict[str, str] = {}
    section: str | None = None
    arg_indent: int | None = None
    current: str | None = None
    for line in lines:
        stripped = line.strip()
        indent = len(line) - len(line.lstrip())
        if indent == 0 and _ARGS_HEADER.match(stripped):
            section, arg_indent, current = "args", None, None
            continue
        if indent == 0 and _SECTION_HEADER.match(stripped):
            section, current = "other", None
            continue
        if section is None:
            description.append(line)
        elif section == "args" and stripped:
            match = _ARG_LINE.match(stripped)
            if match and (arg_indent is None or indent <= arg_indent):
                arg_indent = indent
                current = match.group(1)
                args[current] = match.group(2).strip()
            elif current is not None:
                args[current] = f"{args[current]} {stripped}".strip()
    return "\n".join(description).strip(), args


def _parameters_schema(
    fn: Callable[..., Any], arg_docs: dict[str, str]
) -> tuple[dict[str, Any], dict[str, type[pydantic.BaseModel]]]:
    hints = get_type_hints(fn, include_extras=True)
    defs: dict[str, Any] = {}
    properties: dict[str, Any] = {}
    required: list[str] = []
    models: dict[str, type[pydantic.BaseModel]] = {}
    for param in inspect.signature(fn).parameters.values():
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            continue
        annotation = hints.get(param.name, Any)
        schema = _schema_for(annotation, defs)
        model = _pydantic_model(annotation)
        if model is not None:
            models[param.name] = model
        if param.name in arg_docs and arg_docs[param.name]:
            schema.setdefault("description", arg_docs[param.name])
        if param.default is param.empty:
            required.append(param.name)
        elif param.default is not None and _json_safe(param.default):
            schema.setdefault("default", param.default)
        properties[param.name] = schema
    parameters: dict[str, Any] = {"type": "object", "properties": properties}
    if required:
        parameters["required"] = required
    # Inline model definitions: the harmony renderer and some servers don't resolve $ref.
    return _inline_refs(parameters, defs, ()), models


def _strip_titles(node: Any) -> Any:
    """Drop pydantic's auto-generated titles, which only add noise to the prompt."""
    if isinstance(node, list):
        return [_strip_titles(v) for v in node]
    if not isinstance(node, dict):
        return node
    out: dict[str, Any] = {}
    for key, value in node.items():
        if key == "title" and isinstance(value, str):
            continue
        if key in ("properties", "$defs") and isinstance(value, dict):
            out[key] = {name: _strip_titles(sub) for name, sub in value.items()}
        else:
            out[key] = _strip_titles(value)
    return out


def _inline_refs(node: Any, defs: dict[str, Any], expanding: tuple[str, ...]) -> Any:
    if isinstance(node, list):
        return [_inline_refs(v, defs, expanding) for v in node]
    if not isinstance(node, dict):
        return node
    rest = {k: _inline_refs(v, defs, expanding) for k, v in node.items() if k != "$ref"}
    ref = node.get("$ref")
    if not isinstance(ref, str) or not ref.startswith("#/$defs/"):
        return rest
    name = ref[len("#/$defs/"):]
    if name not in defs or name in expanding:  # unknown or recursive: leave it untyped
        return rest
    return {**_inline_refs(defs[name], defs, expanding + (name,)), **rest}


def _json_safe(value: Any) -> bool:
    try:
        json.dumps(value)
    except (TypeError, ValueError):
        return False
    return True


def _pydantic_model(annotation: Any) -> type[pydantic.BaseModel] | None:
    if get_origin(annotation) is Annotated:
        annotation = get_args(annotation)[0]
    if get_origin(annotation) in (Union, types.UnionType):
        options = [a for a in get_args(annotation) if a is not type(None)]
        annotation = options[0] if len(options) == 1 else None
    if isinstance(annotation, type) and issubclass(annotation, pydantic.BaseModel):
        return annotation
    return None


def _schema_for(annotation: Any, defs: dict[str, Any]) -> dict[str, Any]:
    """JSON schema for `annotation`; model definitions referenced via $ref are collected in `defs`."""
    if annotation is Any or annotation is inspect.Parameter.empty:
        return {}
    origin, args = get_origin(annotation), get_args(annotation)
    if origin is Annotated:
        schema = _schema_for(args[0], defs)
        notes = [a for a in args[1:] if isinstance(a, str)]
        if notes:
            schema["description"] = " ".join(notes)
        return schema
    if annotation in _PRIMITIVES:
        return {"type": _PRIMITIVES[annotation]}
    if annotation is type(None):
        return {"type": "null"}
    if origin in (Union, types.UnionType):
        options = [a for a in args if a is not type(None)]
        if len(options) == 1:
            return _schema_for(options[0], defs)
        return {"anyOf": [_schema_for(a, defs) for a in options]}
    if origin is Literal:
        schema: dict[str, Any] = {"enum": list(args)}
        kinds = {_PRIMITIVES.get(type(a)) for a in args}
        if len(kinds) == 1 and None not in kinds:
            schema["type"] = kinds.pop()
        return schema
    if isinstance(annotation, type) and issubclass(annotation, enum.Enum):
        return {"enum": [member.value for member in annotation]}
    if isinstance(annotation, type) and issubclass(annotation, pydantic.BaseModel):
        schema = _strip_titles(annotation.model_json_schema(ref_template="#/$defs/{model}"))
        defs.update(schema.pop("$defs", {}))
        return schema
    if annotation in (list, tuple, set, frozenset) or origin in (list, tuple, set, frozenset, Sequence):
        items = args[0] if args and args[0] is not Ellipsis else Any
        return {"type": "array", "items": _schema_for(items, defs)}
    if annotation is dict or origin in (dict, Mapping):
        return {"type": "object"}
    return {}
