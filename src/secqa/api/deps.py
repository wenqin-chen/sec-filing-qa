"""Application state and the request-time dependencies built on it.

:class:`AppState` is created once by :func:`secqa.api.app.create_app` and stored on
``app.state.secqa``. Objects that are cheap and key-free (price table, verifier, the default
provider) exist from construction; the index handles (store, embedder, retrievers) are loaded
by :func:`load_index` inside the lifespan so ``/healthz`` answers before the index is opened and
``/readyz`` can explain why the service is not ready instead of the process crash-looping.

Request-time helpers:

* :func:`resolve_provider` -- ``AskRequest.provider`` -> a cached :class:`LLMProvider`. Only
  ``mock``, ``mock:abstain``, ``openai:<model>`` and ``anthropic:<model>`` are accepted from a
  client (``scripted:<path>`` would read a server file named by the caller). When the server
  has ``SECQA_API_KEY`` set, anything but the default provider needs ``X-API-Key``.
* :func:`require_api_key` -- the same gate for ``agent`` mode.
* :func:`effective_cost_cap` / :meth:`AppState.charge` -- per-request cap clamped to the
  server cap and the in-memory, per-instance daily budget (``SECQA_DAILY_BUDGET_USD``).
"""

from __future__ import annotations

import hmac
import importlib
import threading
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import get_args

from fastapi import Request

from secqa.api.errors import Forbidden, NotReady, ProviderUnavailable, Unprocessable
from secqa.core.contracts import Embedder, LLMProvider, RetrievalStrategy
from secqa.core.errors import ConfigError, IndexMismatch, SecqaError
from secqa.core.logging import get_logger
from secqa.core.settings import Settings, provider_vendor
from secqa.embeddings import get_embedder
from secqa.grounding import CitationVerifier
from secqa.indexing import fetch_index
from secqa.providers import DEFAULT_MODELS_YAML, PriceTable, get_provider
from secqa.providers.pricing import FREE_PROVIDERS
from secqa.retrieval import Retriever
from secqa.store import DuckDBStore

log = get_logger(__name__)

API_KEY_HEADER = "X-API-Key"
CLIENT_PROVIDER_VENDORS: frozenset[str] = frozenset({"mock", "openai", "anthropic"})
"""Vendors a client may name in ``AskRequest.provider``; ``scripted`` is test-only."""
STRATEGIES: tuple[str, ...] = get_args(RetrievalStrategy)
DEFAULT_STRATEGY: RetrievalStrategy = "hybrid"
DEFAULT_RETRIEVER_K = 8


def utc_today() -> date:
    """The UTC date used for the daily budget window."""
    return datetime.now(tz=UTC).date()


@dataclass(frozen=True)
class IndexHandles:
    """The three objects every index-backed endpoint needs, guaranteed non-``None``."""

    store: DuckDBStore
    embedder: Embedder
    retriever: Retriever


@dataclass
class AppState:
    """Process-wide service state (one per :class:`FastAPI` app).

    Attributes:
        settings: The settings the app was created with.
        verifier: Citation verifier shared by rag and agent so results are identical.
        prices: Price table (``models.yaml``) used for ``cost_usd`` and budget caps.
        providers: Providers by spec string; the default provider is present from the start,
            client-requested ones are added on first use.
        store / embedder / retriever: Index handles, ``None`` until :func:`load_index` ran.
        retrievers: One :class:`Retriever` per strategy (``bm25`` / ``dense`` / ``hybrid``)
            over the same store and embedder; ``retriever`` is the hybrid one.
        ready: True once the index handles are loaded.
        not_ready_reason: Why ``ready`` is False (shown by ``/readyz``).
        spent_today_usd: Cost charged since ``spend_day`` (UTC); in-memory, per instance.
    """

    settings: Settings
    verifier: CitationVerifier
    prices: PriceTable
    providers: dict[str, LLMProvider]
    store: DuckDBStore | None = None
    embedder: Embedder | None = None
    retriever: Retriever | None = None
    retrievers: dict[str, Retriever] = field(default_factory=dict)
    ready: bool = False
    not_ready_reason: str | None = "index not loaded yet"
    spent_today_usd: float = 0.0
    spend_day: date = field(default_factory=utc_today)
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    # ---- index handles --------------------------------------------------------------------

    def require_index(self) -> IndexHandles:
        """Return the index handles or raise :class:`NotReady` (503) with the reason."""
        if not self.ready or self.store is None or self.embedder is None or self.retriever is None:
            raise NotReady(self.not_ready_reason or "index not loaded")
        return IndexHandles(store=self.store, embedder=self.embedder, retriever=self.retriever)

    def retriever_for(self, strategy: str) -> Retriever:
        """The retriever for ``strategy`` (``NotReady`` when the index is not loaded)."""
        self.require_index()
        try:
            return self.retrievers[strategy]
        except KeyError:
            raise Unprocessable(
                f"unknown strategy {strategy!r}; expected one of {', '.join(STRATEGIES)}"
            ) from None

    @property
    def default_provider(self) -> LLMProvider:
        """The provider named by ``settings.provider`` (constructed at app creation)."""
        return self.providers[self.settings.provider]

    # ---- daily budget ---------------------------------------------------------------------

    def charge(self, cost_usd: float) -> float:
        """Add ``cost_usd`` to today's spend (window rolls at UTC midnight); return the total."""
        if cost_usd < 0:
            raise ValueError(f"cost_usd must be >= 0, got {cost_usd!r}")
        with self.lock:
            self._roll_day()
            self.spent_today_usd += cost_usd
            return self.spent_today_usd

    def daily_budget_left(self) -> float:
        """USD still available under ``settings.daily_budget_usd`` for today (never negative)."""
        with self.lock:
            self._roll_day()
            return max(0.0, self.settings.daily_budget_usd - self.spent_today_usd)

    def _roll_day(self) -> None:
        today = utc_today()
        if today != self.spend_day:
            log.info("daily_budget_reset", previous_day=self.spend_day.isoformat())
            self.spend_day = today
            self.spent_today_usd = 0.0


# ---- construction and lifecycle ------------------------------------------------------------


def build_state(settings: Settings) -> AppState:
    """Create the state with everything that needs no index: prices, verifier, default provider.

    Raises:
        ConfigError: the default provider needs a vendor key that is not configured
            (``Settings.validate_provider_keys``), or the price table is malformed.
    """
    settings.validate_provider_keys()
    provider = get_provider(settings.provider, settings)
    prices = load_prices()
    if not prices.has(provider.provider, provider.model):
        raise ConfigError(
            f"default provider {settings.provider!r} resolves to "
            f"{provider.provider}:{provider.model}, which is not priced in models.yaml"
        )
    return AppState(
        settings=settings,
        verifier=CitationVerifier(),
        prices=prices,
        providers={settings.provider: provider},
    )


def load_prices(path: Path = DEFAULT_MODELS_YAML) -> PriceTable:
    """Load ``models.yaml``; an absent file yields an empty table (free providers still price).

    The eval module owns the file. Without it every paid provider is refused with a clear
    503 rather than silently reported as ``$0``.
    """
    if Path(path).is_file():
        return PriceTable.load(Path(path))
    log.warning("models_yaml_missing", path=str(path), effect="only mock providers can be priced")
    return PriceTable({}, as_of=utc_today())


def load_index(state: AppState) -> None:
    """Open the index read-only and build the embedder and retrievers; never raises.

    Order: fetch ``settings.index_url`` when the DuckDB file is absent, open the file read-only
    (serving never writes; ingest is CLI-only), build the embedder from ``settings.embedder``
    and one retriever per strategy (each validates embedder/store compatibility). Any failure
    leaves ``state.ready`` False with the reason in ``state.not_ready_reason``.

    One store serves every request: sync endpoints run on the threadpool, and
    :attr:`DuckDBStore.conn` gives each thread its own cursor, so concurrent requests never
    share a DuckDB result set.
    """
    settings = state.settings
    path = Path(settings.duckdb_path)
    try:
        if not path.is_file() and settings.index_url:
            log.info("index_fetch_started", url=settings.index_url, dest=str(path))
            fetch_index(settings.index_url, path)
        if not path.is_file():
            raise ConfigError(
                f"index not found at {path}; run `secqa ingest ...` or set SECQA_INDEX_URL"
            )
        embedder = get_embedder(settings.embedder, settings)
        store = DuckDBStore(
            path,
            embed_dim=embedder.dim,
            read_only=True,
            memory_limit=settings.duckdb_memory_limit,
            threads=settings.duckdb_threads,
        )
        try:
            retrievers = {
                strategy: Retriever(store, embedder, strategy=strategy, k=DEFAULT_RETRIEVER_K)  # type: ignore[arg-type]  # strategy comes from the RetrievalStrategy literal
                for strategy in STRATEGIES
            }
        except (IndexMismatch, ConfigError):
            store.close()
            raise
    except (SecqaError, OSError, ValueError) as exc:
        state.ready = False
        state.not_ready_reason = f"{type(exc).__name__}: {exc}"
        log.error("index_load_failed", reason=state.not_ready_reason)
        return
    state.store = store
    state.embedder = embedder
    state.retrievers = retrievers
    state.retriever = retrievers[DEFAULT_STRATEGY]
    state.ready = True
    state.not_ready_reason = None
    counts = store.counts()
    log.info(
        "index_loaded",
        path=str(path),
        embedder=embedder.name,
        dim=embedder.dim,
        bm25_backend=store.bm25_backend,
        **counts,
    )


def close_index(state: AppState) -> None:
    """Release the DuckDB handle (idempotent); called from the lifespan on shutdown."""
    state.ready = False
    state.not_ready_reason = "shutting down"
    state.retrievers = {}
    state.retriever = None
    state.embedder = None
    if state.store is not None:
        state.store.close()
        state.store = None


# ---- request-time dependencies --------------------------------------------------------------


def get_state(request: Request) -> AppState:
    """FastAPI dependency: the :class:`AppState` of the app serving ``request``."""
    state = getattr(request.app.state, "secqa", None)
    if not isinstance(state, AppState):
        raise NotReady("application state missing: was the app built with create_app()?")
    return state


def api_key_matches(state: AppState, presented: str | None) -> bool:
    """Constant-time comparison of a presented ``X-API-Key`` against ``SECQA_API_KEY``."""
    configured = state.settings.api_key
    if configured is None or not configured.get_secret_value():
        return False
    if not presented:
        return False
    return hmac.compare_digest(presented.encode("utf-8"), configured.get_secret_value().encode())


def require_api_key(state: AppState, presented: str | None, *, what: str) -> None:
    """Raise :class:`Forbidden` when the server has an API key and ``presented`` is not it.

    With no ``SECQA_API_KEY`` configured every caller is trusted (local development, CI).
    """
    configured = state.settings.api_key
    if configured is None or not configured.get_secret_value():
        return
    if not api_key_matches(state, presented):
        raise Forbidden(f"{what} requires a valid {API_KEY_HEADER} header on this server")


def resolve_provider(state: AppState, spec: str | None, presented_key: str | None) -> LLMProvider:
    """``AskRequest.provider`` -> provider instance (cached in ``state.providers``).

    Raises:
        Unprocessable: the spec names a vendor a client may not use, or a model with no price.
        Forbidden: a non-default provider was requested without the server's API key.
        ProviderUnavailable: the vendor key for the spec is not configured on the server.
    """
    requested = (spec or "").strip() or state.settings.provider
    if requested == state.settings.provider:
        return state.default_provider
    require_api_key(state, presented_key, what=f"provider {requested!r}")
    cached = state.providers.get(requested)
    if cached is not None:
        return cached
    try:
        vendor = provider_vendor(requested)
    except ConfigError as exc:
        raise Unprocessable(str(exc)) from exc
    if vendor not in CLIENT_PROVIDER_VENDORS:
        raise Unprocessable(
            f"unknown provider {requested!r}; expected 'mock', 'mock:abstain', "
            "'openai:<model>' or 'anthropic:<model>'"
        )
    try:
        state.settings.validate_provider_spec(requested)
    except ConfigError as exc:
        raise ProviderUnavailable(str(exc)) from exc
    try:
        provider = get_provider(requested, state.settings)
    except ConfigError as exc:
        raise Unprocessable(str(exc)) from exc
    if not state.prices.has(provider.provider, provider.model):
        raise Unprocessable(
            f"model {provider.provider}:{provider.model} is not priced in models.yaml "
            f"(as_of {state.prices.as_of.isoformat()}); it cannot be billed safely"
        )
    state.providers[requested] = provider
    log.info("provider_added", spec=requested, provider=provider.provider, model=provider.model)
    return provider


def is_paid(provider: LLMProvider) -> bool:
    """True for providers that bill tokens (everything but ``mock`` / ``scripted``)."""
    return provider.provider.lower() not in FREE_PROVIDERS


def effective_cost_cap(state: AppState, requested: float | None) -> float:
    """The per-request USD cap: the client's value clamped to ``settings.max_cost_usd``."""
    server_cap = state.settings.max_cost_usd
    if requested is None:
        return server_cap
    return min(requested, server_cap)


def resolve_judge_version() -> str | None:
    """``JUDGE_VERSION`` of the eval module's judge, or ``None`` when it is not importable.

    The API must not hard-depend on the evaluation harness (it is not needed to serve), so
    this is a guarded lazy import; ``/version`` reports ``null`` when it is absent.
    """
    try:
        module = importlib.import_module("secqa.eval.judge")
    except ImportError:
        return None
    value = getattr(module, "JUDGE_VERSION", None)
    return str(value) if value else None


__all__ = [
    "API_KEY_HEADER",
    "CLIENT_PROVIDER_VENDORS",
    "DEFAULT_RETRIEVER_K",
    "DEFAULT_STRATEGY",
    "AppState",
    "IndexHandles",
    "api_key_matches",
    "build_state",
    "close_index",
    "effective_cost_cap",
    "get_state",
    "is_paid",
    "load_index",
    "load_prices",
    "require_api_key",
    "resolve_judge_version",
    "resolve_provider",
    "utc_today",
]
