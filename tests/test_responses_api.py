import time

import pytest
from fastapi.testclient import TestClient
from openai_harmony import (
    HarmonyEncodingName,
    HarmonyError,
    load_harmony_encoding,
)

from gpt_oss.responses_api.api_server import create_api_server
from gpt_oss.responses_api.events import (
    ResponseWebSearchCallCompleted,
    ResponseWebSearchCallInProgress,
    ResponseWebSearchCallSearching,
)

try:
    encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
except HarmonyError:
    pytest.skip(
        "Harmony encoding vocabulary not available in test environment.",
        allow_module_level=True,
    )

fake_tokens = encoding.encode(
    "<|channel|>final<|message|>Hey there<|return|>", allowed_special="all"
)

token_queue = fake_tokens.copy()


def stub_infer_next_token(
    tokens: list[int], temperature: float = 0.0, new_request: bool = False
) -> int:
    global token_queue
    next_tok = token_queue.pop(0)
    if len(token_queue) == 0:
        token_queue = fake_tokens.copy()
    time.sleep(0.1)
    return next_tok


@pytest.fixture
def test_client():
    return TestClient(
        create_api_server(infer_next_token=stub_infer_next_token, encoding=encoding)
    )


def test_health_check(test_client):
    response = test_client.post(
        "/v1/responses",
        json={
            "model": "gpt-oss-120b",
            "input": "Hello, world!",
        },
    )
    print(response.json())
    assert response.status_code == 200


def test_web_search_call_events_include_item_id():
    web_search_call_id = "ws_test_id"
    events = [
        ResponseWebSearchCallInProgress(
            type="response.web_search_call.in_progress",
            output_index=0,
            item_id=web_search_call_id,
        ),
        ResponseWebSearchCallSearching(
            type="response.web_search_call.searching",
            output_index=0,
            item_id=web_search_call_id,
        ),
        ResponseWebSearchCallCompleted(
            type="response.web_search_call.completed",
            output_index=0,
            item_id=web_search_call_id,
        ),
    ]

    for event in events:
        sse_payload = f"event: {event.type}\ndata: {event.model_dump_json(indent=None)}\n\n"
        assert "\"item_id\":" in sse_payload
        assert web_search_call_id in sse_payload
