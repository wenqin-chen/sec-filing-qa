"""OpenAIEmbedder over respx-mocked HTTP: request shape, ordering, usage, retries, errors."""

from __future__ import annotations

import json
from typing import Any

import httpx
import numpy as np
import pytest
import respx

from secqa.core.contracts import Embedder
from secqa.core.errors import ConfigError, ProviderError
from secqa.embeddings import OpenAIEmbedder

URL = "https://api.openai.com/v1/embeddings"
KEY = "sk-test-not-a-real-key"


def _embedder(**kwargs: Any) -> OpenAIEmbedder:
    defaults: dict[str, Any] = {"dim": 4, "api_key": KEY, "backoff_s": 0.0}
    defaults.update(kwargs)
    return OpenAIEmbedder(**defaults)


def test_requires_key() -> None:
    with pytest.raises(ConfigError, match="OPENAI_API_KEY"):
        OpenAIEmbedder()
    with pytest.raises(ConfigError):
        OpenAIEmbedder(api_key="  ")
    with pytest.raises(ConfigError):
        OpenAIEmbedder(api_key=KEY, dim=0)


def test_defaults() -> None:
    with OpenAIEmbedder(api_key=KEY) as embedder:
        assert isinstance(embedder, Embedder)
        assert embedder.name == "text-embedding-3-small"
        assert embedder.model == "text-embedding-3-small"
        assert embedder.dim == 384


def test_request_body_ordering_normalisation_and_usage(
    respx_router: respx.MockRouter, openai_embeddings_payload: dict[str, Any]
) -> None:
    route = respx_router.post(URL).mock(
        return_value=httpx.Response(200, json=openai_embeddings_payload)
    )
    seen: list[int] = []
    with _embedder(on_usage=seen.append) as embedder:
        vectors = embedder.embed(["alpha", "beta", "gamma"], kind="query")

    assert route.call_count == 1
    request = route.calls.last.request
    assert request.headers["authorization"] == f"Bearer {KEY}"
    assert request.headers["user-agent"].startswith("secqa/")
    body = json.loads(request.content)
    assert body == {
        "model": "text-embedding-3-small",
        "input": ["alpha", "beta", "gamma"],
        "encoding_format": "float",
        "dimensions": 4,
    }
    # Fixture lists index 1 first: rows must follow ``index``, not payload order.
    expected = np.array(
        [[1.0, 0.0, 0.0, 0.0], [0.0, 0.6, 0.8, 0.0], [0.5, 0.5, 0.5, 0.5]], dtype=np.float32
    )
    np.testing.assert_allclose(vectors, expected, atol=1e-6)
    assert vectors.dtype == np.float32
    assert seen == [21]
    assert embedder.total_tokens == 21


def test_batches_and_blank_texts(
    respx_router: respx.MockRouter, openai_embeddings_payload: dict[str, Any]
) -> None:
    def responder(request: httpx.Request) -> httpx.Response:
        inputs = json.loads(request.content)["input"]
        data = [
            {"object": "embedding", "index": i, "embedding": [float(len(text)), 1.0, 0.0, 0.0]}
            for i, text in enumerate(inputs)
        ]
        body = {"object": "list", "data": data, "usage": {"total_tokens": 5}}
        return httpx.Response(200, json=body)

    route = respx_router.post(URL).mock(side_effect=responder)
    seen: list[int] = []
    with _embedder(on_usage=seen.append) as embedder:
        vectors = embedder.embed(["a", "", "bb", "ccc", "   "], batch_size=2)

    # 3 non-blank texts, batch_size 2 -> 2 requests; blanks never sent, zero rows in place.
    assert route.call_count == 2
    sent = [json.loads(call.request.content)["input"] for call in route.calls]
    assert sent == [["a", "bb"], ["ccc"]]
    assert vectors.shape == (5, 4)
    assert not vectors[1].any() and not vectors[4].any()
    np.testing.assert_allclose(np.linalg.norm(vectors[[0, 2, 3]], axis=1), 1.0, atol=1e-6)
    assert seen == [5, 5]
    assert embedder.total_tokens == 10


def test_retries_on_429_then_succeeds(
    respx_router: respx.MockRouter, openai_embeddings_payload: dict[str, Any]
) -> None:
    route = respx_router.post(URL).mock(
        side_effect=[
            httpx.Response(429, json={"error": {"message": "rate limited"}}),
            httpx.Response(503, text="upstream busy"),
            httpx.Response(200, json=openai_embeddings_payload),
        ]
    )
    with _embedder(max_attempts=4) as embedder:
        vectors = embedder.embed(["alpha", "beta", "gamma"])
    assert route.call_count == 3
    assert vectors.shape == (3, 4)


def test_retry_budget_exhausted_raises_retryable(respx_router: respx.MockRouter) -> None:
    route = respx_router.post(URL).mock(return_value=httpx.Response(500, text="boom"))
    with _embedder(max_attempts=2) as embedder, pytest.raises(ProviderError) as info:
        embedder.embed(["alpha"])
    assert route.call_count == 2
    assert info.value.retryable is True
    assert info.value.provider == "openai"
    assert "HTTP 500" in str(info.value)


def test_network_error_is_retryable(respx_router: respx.MockRouter) -> None:
    route = respx_router.post(URL).mock(side_effect=httpx.ConnectError("no route"))
    with _embedder(max_attempts=3) as embedder, pytest.raises(ProviderError) as info:
        embedder.embed(["alpha"])
    assert route.call_count == 3
    assert info.value.retryable is True


def test_401_is_not_retried(respx_router: respx.MockRouter) -> None:
    route = respx_router.post(URL).mock(
        return_value=httpx.Response(401, json={"error": {"message": "Incorrect API key"}})
    )
    with _embedder(max_attempts=4) as embedder, pytest.raises(ProviderError) as info:
        embedder.embed(["alpha"])
    assert route.call_count == 1
    assert info.value.retryable is False
    assert "Incorrect API key" in str(info.value)


def test_dimension_mismatch_is_provider_error(
    respx_router: respx.MockRouter, openai_embeddings_payload: dict[str, Any]
) -> None:
    respx_router.post(URL).mock(return_value=httpx.Response(200, json=openai_embeddings_payload))
    with _embedder(dim=384) as embedder, pytest.raises(ProviderError, match="dim=384"):
        embedder.embed(["alpha", "beta", "gamma"])


@pytest.mark.parametrize(
    "payload",
    [
        {"data": [{"index": 0, "embedding": [1, 0, 0, 0]}]},  # too few rows
        {  # duplicate index
            "data": [
                {"index": 0, "embedding": [1, 0, 0, 0]},
                {"index": 0, "embedding": [0, 1, 0, 0]},
            ]
        },
        {"data": [{"embedding": [1, 0, 0, 0]}, {"index": 1, "embedding": [0, 1, 0, 0]}]},
        {"data": "nope"},
        [],
    ],
)
def test_malformed_payloads(respx_router: respx.MockRouter, payload: Any) -> None:
    respx_router.post(URL).mock(return_value=httpx.Response(200, json=payload))
    with _embedder() as embedder, pytest.raises(ProviderError) as info:
        embedder.embed(["alpha", "beta"])
    assert info.value.retryable is False


def test_non_json_body(respx_router: respx.MockRouter) -> None:
    respx_router.post(URL).mock(return_value=httpx.Response(200, text="<html>gateway</html>"))
    with _embedder() as embedder, pytest.raises(ProviderError, match="non-JSON"):
        embedder.embed(["alpha"])


def test_legacy_model_omits_dimensions(respx_router: respx.MockRouter) -> None:
    payload = {"data": [{"index": 0, "embedding": [1, 0, 0, 0]}], "usage": {"total_tokens": 1}}
    route = respx_router.post(URL).mock(return_value=httpx.Response(200, json=payload))
    with _embedder(model="text-embedding-ada-002") as embedder:
        embedder.embed(["alpha"])
    assert "dimensions" not in json.loads(route.calls.last.request.content)


def test_custom_base_url(respx_router: respx.MockRouter) -> None:
    payload = {"data": [{"index": 0, "embedding": [1, 0, 0, 0]}], "usage": {"total_tokens": 1}}
    route = respx_router.post("https://gateway.example.com/openai/v1/embeddings").mock(
        return_value=httpx.Response(200, json=payload)
    )
    with _embedder(base_url="https://gateway.example.com/openai/v1/") as embedder:
        embedder.embed(["alpha"])
    assert route.call_count == 1


def test_empty_input_makes_no_request(respx_router: respx.MockRouter) -> None:
    route = respx_router.post(URL).mock(return_value=httpx.Response(500))
    with _embedder() as embedder:
        assert embedder.embed([]).shape == (0, 4)
        assert embedder.embed(["", " "]).shape == (2, 4)
    assert route.call_count == 0
