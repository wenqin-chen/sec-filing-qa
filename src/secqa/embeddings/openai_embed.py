"""OpenAI embeddings over plain ``httpx`` (``POST /v1/embeddings``).

Deliberately *not* built on the ``openai`` SDK: per project rules only ``secqa.providers`` imports
vendor SDKs, and the embeddings endpoint is a single JSON POST that ``httpx`` + ``respx`` cover
completely in tests. ``text-embedding-3-*`` models accept a ``dimensions`` parameter, which is how
the default 384-d output matches the local ``bge-small`` width so both fit the same DuckDB column.

Cost accounting: the endpoint reports ``usage.total_tokens`` per request; every value is passed to
``on_usage`` so the indexing pipeline can price an ingest run from ``models.yaml``.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import numpy as np
from tenacity import (
    Retrying,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
)

from secqa import __version__
from secqa.core.errors import ConfigError, ProviderError
from secqa.core.logging import get_logger
from secqa.core.settings import get_settings
from secqa.embeddings.base import BaseEmbedder, EmbedKind, UsageCallback

_log = get_logger(__name__)

DEFAULT_OPENAI_MODEL = "text-embedding-3-small"
DEFAULT_BASE_URL = "https://api.openai.com/v1"
_RETRYABLE_STATUS = frozenset({408, 409, 429, 500, 502, 503, 504})
_PROVIDER = "openai"


def _is_retryable(exc: BaseException) -> bool:
    return isinstance(exc, ProviderError) and exc.retryable


class OpenAIEmbedder(BaseEmbedder):
    """``text-embedding-3-*`` embedder with the ``dimensions`` parameter and usage callback.

    Args:
        model: OpenAI embedding model id.
        dim: Requested output width. Sent as ``dimensions`` for ``text-embedding-3-*`` models;
            for older models it must equal the model's native width (checked on the response).
        api_key: Explicit key; ``None`` reads ``OPENAI_API_KEY`` via :func:`get_settings`.
        on_usage: Called with ``usage.total_tokens`` after every successful request.
        base_url: API root (override for Azure OpenAI-compatible gateways / test servers).
        timeout_s: Per-request timeout.
        max_attempts: Total attempts for retryable failures (429, 5xx, network, timeout).
        backoff_s: Initial exponential backoff; ``0`` disables waiting (tests).

    Raises:
        ConfigError: No API key available or invalid ``dim``.
    """

    def __init__(
        self,
        model: str = DEFAULT_OPENAI_MODEL,
        dim: int = 384,
        api_key: str | None = None,
        on_usage: UsageCallback | None = None,
        *,
        base_url: str = DEFAULT_BASE_URL,
        timeout_s: float = 30.0,
        max_attempts: int = 4,
        backoff_s: float = 0.5,
    ) -> None:
        if not model or not model.strip():
            raise ConfigError("openai embedder model must not be empty")
        if not isinstance(dim, int) or isinstance(dim, bool) or dim < 1:
            raise ConfigError(f"openai embedder dim must be a positive int, got {dim!r}")
        if max_attempts < 1:
            raise ConfigError(f"max_attempts must be >= 1, got {max_attempts}")
        key = api_key
        if key is None:
            secret = get_settings().openai_api_key
            key = secret.get_secret_value() if secret is not None else None
        if key is None or not key.strip():
            raise ConfigError("the openai embedder requires OPENAI_API_KEY but it is not set")

        self.model = model.strip()
        self.name = self.model
        self.dim = dim
        self.on_usage = on_usage
        self.total_tokens = 0  # cumulative billable tokens seen by this instance
        self.base_url = base_url.rstrip("/")
        self._send_dimensions = self.model.startswith("text-embedding-3")
        self._client = httpx.Client(
            base_url=self.base_url,
            timeout=timeout_s,
            headers={
                "Authorization": f"Bearer {key.strip()}",
                "Content-Type": "application/json",
                "User-Agent": f"secqa/{__version__}",
            },
        )
        self._retrying = Retrying(
            stop=stop_after_attempt(max_attempts),
            wait=wait_exponential_jitter(initial=backoff_s, max=backoff_s * 16, jitter=backoff_s),
            retry=retry_if_exception(_is_retryable),
            reraise=True,
        )
        _log.debug("openai embedder ready", embedder=self.name, dim=dim, base_url=self.base_url)

    # ---- protocol ----

    def _embed_batch(self, texts: list[str], kind: EmbedKind) -> np.ndarray:
        """POST one batch; ``kind`` is ignored (OpenAI embeddings are symmetric)."""
        body: dict[str, Any] = {"model": self.model, "input": texts, "encoding_format": "float"}
        if self._send_dimensions:
            body["dimensions"] = self.dim
        payload = self._retrying(self._post, body)
        return self._parse(payload, expected=len(texts))

    # ---- HTTP ----

    def _post(self, body: dict[str, Any]) -> dict[str, Any]:
        try:
            response = self._client.post("/embeddings", json=body)
        except httpx.TimeoutException as exc:
            raise ProviderError(f"timeout: {exc}", retryable=True, provider=_PROVIDER) from exc
        except httpx.TransportError as exc:
            raise ProviderError(
                f"network error: {exc}", retryable=True, provider=_PROVIDER
            ) from exc

        if response.status_code >= 400:
            raise ProviderError(
                f"HTTP {response.status_code} from embeddings endpoint: {_error_message(response)}",
                retryable=response.status_code in _RETRYABLE_STATUS,
                provider=_PROVIDER,
            )
        try:
            payload = response.json()
        except (json.JSONDecodeError, ValueError) as exc:
            raise ProviderError(
                "embeddings endpoint returned non-JSON body", retryable=False, provider=_PROVIDER
            ) from exc
        if not isinstance(payload, dict):
            raise ProviderError(
                "embeddings endpoint returned a non-object JSON body",
                retryable=False,
                provider=_PROVIDER,
            )
        return payload

    def _parse(self, payload: dict[str, Any], *, expected: int) -> np.ndarray:
        data = payload.get("data")
        if not isinstance(data, list) or len(data) != expected:
            got = len(data) if isinstance(data, list) else type(data).__name__
            raise ProviderError(
                f"expected {expected} embeddings, got {got}", retryable=False, provider=_PROVIDER
            )
        # The API documents ``index`` as the position in the input; never trust list order.
        try:
            ordered = sorted(data, key=lambda item: int(item["index"]))
            vectors = np.asarray([item["embedding"] for item in ordered], dtype=np.float32)
        except (KeyError, TypeError, ValueError) as exc:
            raise ProviderError(
                f"malformed embeddings payload: {exc}", retryable=False, provider=_PROVIDER
            ) from exc
        if [int(item["index"]) for item in ordered] != list(range(expected)):
            raise ProviderError(
                "embeddings payload indices are not 0..n-1", retryable=False, provider=_PROVIDER
            )
        if vectors.ndim != 2 or vectors.shape[1] != self.dim:
            width = vectors.shape[1] if vectors.ndim == 2 else "?"
            raise ProviderError(
                f"model {self.model!r} returned {width}-d vectors but dim={self.dim} was requested",
                retryable=False,
                provider=_PROVIDER,
            )

        usage = payload.get("usage") or {}
        tokens = usage.get("total_tokens", usage.get("prompt_tokens", 0))
        tokens = int(tokens) if isinstance(tokens, int | float) else 0
        self.total_tokens += tokens
        if self.on_usage is not None:
            self.on_usage(tokens)
        _log.debug("openai embeddings", embedder=self.name, n=expected, tokens=tokens)
        return vectors

    # ---- lifecycle ----

    def close(self) -> None:
        """Close the underlying HTTP client (safe to call more than once)."""
        self._client.close()

    def __enter__(self) -> OpenAIEmbedder:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


def _error_message(response: httpx.Response) -> str:
    """Best-effort ``error.message`` from an OpenAI error body, else a truncated raw body."""
    try:
        body = response.json()
        message = body["error"]["message"]
        if isinstance(message, str):
            return message
    except (ValueError, KeyError, TypeError):
        pass
    return response.text[:200]


__all__ = ["DEFAULT_BASE_URL", "DEFAULT_OPENAI_MODEL", "OpenAIEmbedder"]
