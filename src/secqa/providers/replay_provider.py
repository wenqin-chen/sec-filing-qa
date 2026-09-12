"""Record / replay cassettes so every paid LLM call can be re-scored with no keys.

One cassette entry is one JSON file ``<cache_dir>/<key>.json`` where ``key`` is
``sha256(provider, model, system, messages, tools, json_schema, max_tokens, effort)`` (see
:func:`secqa.providers.base.request_key`). The file stores the canonical request (for debugging)
and the :class:`LLMResponse` verbatim. The stored request is the full prompt, so a cassette
recorded from a FinanceBench run contains dataset text (question, reference answer,
justification, gold evidence pages); ``cassettes/README.md`` states the distribution policy.

Modes:

* ``record`` - read-through cache: serve a hit from disk, otherwise call the inner provider and
  write the entry. Nothing is paid twice, and a resumed run reuses earlier answers.
* ``replay`` - serve hits, raise :class:`CassetteMiss` on a miss (never touches the network).
* ``off`` - pass every call straight through.

Responses served from disk carry ``cached=True`` and keep the recorded ``latency_ms`` so that
timing metrics from a replay are recognisably not fresh measurements.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from secqa.core.contracts import LLMProvider, LLMResponse, Message, ToolSpec
from secqa.core.errors import CassetteMiss, ConfigError
from secqa.core.logging import get_logger
from secqa.providers.base import BaseProvider, Effort, canonical_request, request_key

CacheMode = Literal["record", "replay", "off"]
_MODES: frozenset[str] = frozenset({"record", "replay", "off"})

log = get_logger("secqa.providers.replay")


class ReplayCacheProvider(BaseProvider):
    """Wrap any :class:`LLMProvider` with an on-disk cassette cache."""

    def __init__(self, inner: LLMProvider, cache_dir: Path, mode: CacheMode):
        if mode not in _MODES:
            raise ConfigError(f"cassette mode must be one of {sorted(_MODES)}, got {mode!r}")
        self.inner = inner
        # provider / model mirror the wrapped provider so callers and price tables see the real
        # vendor (both are fixed at construction on every provider).
        self.provider = inner.provider
        self.model = inner.model
        self.cache_dir = Path(cache_dir)
        self.mode: CacheMode = mode
        self.hits = 0
        self.misses = 0

    def params(self) -> dict[str, Any]:
        """Delegate to the inner provider (the wrapper adds no sampling parameters)."""
        params = getattr(self.inner, "params", None)
        return dict(params()) if callable(params) else {}

    def key_for(
        self,
        messages: list[Message],
        *,
        system: str = "",
        tools: list[ToolSpec] | None = None,
        json_schema: dict[str, Any] | None = None,
        max_tokens: int = 2048,
        effort: Effort | None = None,
    ) -> str:
        """Cassette key for a request (exposed so tests and ``rescore`` can inspect it)."""
        return request_key(
            canonical_request(
                self.provider,
                self.model,
                messages,
                system=system,
                tools=tools,
                json_schema=json_schema,
                max_tokens=max_tokens,
                effort=effort,
            )
        )

    def path_for(self, key: str) -> Path:
        """File that stores the entry for ``key``."""
        return self.cache_dir / f"{key}.json"

    def complete(
        self,
        messages: list[Message],
        *,
        system: str = "",
        tools: list[ToolSpec] | None = None,
        json_schema: dict[str, Any] | None = None,
        max_tokens: int = 2048,
        effort: Effort | None = None,
    ) -> LLMResponse:
        """Serve from the cassette when possible; otherwise call ``inner`` (and record)."""
        call_kwargs: dict[str, Any] = {
            "system": system,
            "tools": tools,
            "json_schema": json_schema,
            "max_tokens": max_tokens,
            "effort": effort,
        }
        if self.mode == "off":
            return self.inner.complete(messages, **call_kwargs)

        request = canonical_request(self.provider, self.model, messages, **call_kwargs)
        key = request_key(request)
        cached = self._read(key)
        if cached is not None:
            self.hits += 1
            log.info("cassette_hit", key=key, mode=self.mode, provider=self.provider)
            return cached
        self.misses += 1
        if self.mode == "replay":
            log.warning("cassette_miss", key=key, provider=self.provider, model=self.model)
            raise CassetteMiss(key)

        response = self.inner.complete(messages, **call_kwargs)
        self._write(key, request, response)
        log.info("cassette_recorded", key=key, provider=self.provider)
        return response

    # ---- storage ----

    def _read(self, key: str) -> LLMResponse | None:
        path = self.path_for(key)
        if not path.is_file():
            return None
        try:
            entry = json.loads(path.read_text(encoding="utf-8"))
            return LLMResponse.model_validate({**entry["response"], "cached": True})
        except (ValueError, KeyError, TypeError) as exc:
            raise ConfigError(f"corrupt cassette entry {path}: {exc}") from exc

    def _write(self, key: str, request: dict[str, Any], response: LLMResponse) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        entry = {
            "key": key,
            "request": request,
            "response": response.model_dump(mode="json"),
        }
        tmp = self.path_for(key).with_suffix(".json.tmp")
        tmp.write_text(json.dumps(entry, ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(self.path_for(key))

    def __repr__(self) -> str:
        return (
            f"ReplayCacheProvider(mode={self.mode!r}, dir={str(self.cache_dir)!r}, "
            f"inner={self.inner!r})"
        )


__all__ = ["CacheMode", "ReplayCacheProvider"]
