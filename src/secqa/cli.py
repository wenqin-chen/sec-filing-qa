"""``secqa``: the Typer command line, the single entry point for every reproducible step.

Every command composes the library modules rather than re-implementing them (SPEC 12: "agents
implement; author reviews, labels, deploys"):

* ``doctor`` -- what this environment can do (keys, model ids, User-Agent, DuckDB FTS, index).
* ``data financebench`` / ``data companies-xbrl`` -- the dataset, its PDFs and companyfacts.
* ``ingest financebench`` / ``ingest ticker`` -- build or extend ``data/index.duckdb``.
* ``index pack | fetch | manifest`` -- the ``index-*.tar.zst`` release asset and its provenance.
* ``ask`` -- one question through rag / agent / closed_book with verified citations.
* ``eval`` / ``rescore`` / ``report`` -- the FinanceBench harness and ``RESULTS.md``.
* ``serve`` -- uvicorn over :func:`secqa.api.app.create_app`.
* ``export`` -- Parquet copies of every table.

Conventions: settings come from the environment / ``.env``
(:func:`secqa.core.settings.get_settings`) and a few flags override them per command
(``--db``, ``--provider``, ``--embedder``); every failure of our own
(:class:`~secqa.core.errors.SecqaError`, a missing file) prints one ``error:`` line to stderr
and exits 1 -- no tracebacks for expected problems; ``--json`` on the commands that produce
data prints plain JSON to stdout for scripts. EDGAR and Hugging Face are touched
only by ``data`` and ``ingest`` (never at answer time), and nothing here ever writes dataset
text under ``results/``.
"""

from __future__ import annotations

import functools
import importlib.util
import json
import logging
import os
import sys
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Annotated, Any, Literal, TypeVar, get_args

import httpx
import typer
from rich.console import Console
from rich.table import Table

from secqa import __version__
from secqa.core.contracts import Answer, Embedder, Mode, RetrievalFilters, RetrievalStrategy
from secqa.core.errors import ConfigError, SecqaError
from secqa.core.logging import configure_logging, get_logger
from secqa.core.settings import EMAIL_RE, Settings, get_settings, provider_vendor
from secqa.edgar import EdgarClient
from secqa.edgar.client import DEFAULT_CACHE_DIR as EDGAR_CACHE_DIR
from secqa.eval.financebench import (
    DEFAULT_CACHE_DIR,
    DEFAULT_SPLIT,
    dataset_revision,
    download_pdfs,
    load_financebench,
    load_questions_jsonl,
)
from secqa.eval.report import render_results_md
from secqa.eval.rescore import read_run_config, rescore
from secqa.eval.runner import (
    DEFAULT_OUT_DIR,
    EvalConfig,
    build_fixture_index,
    load_config,
    run_eval,
)
from secqa.indexing import (
    DEFAULT_COMPANIES_PATH,
    build_manifest,
    fetch_index,
    ingest_financebench_corpus,
    ingest_ticker,
    load_companies,
    load_xbrl_for_companies,
    pack_index,
)
from secqa.providers import DEFAULT_MODELS_YAML, PriceTable, get_provider
from secqa.store import DuckDBStore, resolve_git_sha

log = get_logger(__name__)

T = TypeVar("T")

MODES: tuple[str, ...] = get_args(Mode)
ASK_MODES: tuple[str, ...] = ("rag", "agent", "closed_book")  # oracle needs gold pages: eval only
STRATEGIES: tuple[str, ...] = get_args(RetrievalStrategy)
DEFAULT_PDF_DIR = DEFAULT_CACHE_DIR / "pdfs"
DEFAULT_INDEX_TARBALL = Path("data") / f"index-v{__version__}.tar.zst"
DEFAULT_PARQUET_DIR = Path("data/parquet")
DEFAULT_CONFIGS_DIR = Path("configs")
DEFAULT_RESULTS_MD = Path("RESULTS.md")
AGENT_TOOL_RESULT_CHARS = 20_000  # CONTRACTS rule 12: room for three full pages per get_pages
VENDOR_CHECK_TIMEOUT_S = 10.0
OPENAI_MODELS_URL = "https://api.openai.com/v1/models/"
ANTHROPIC_MODELS_URL = "https://api.anthropic.com/v1/models/"
ANTHROPIC_VERSION_HEADER = "2023-06-01"

out = Console()
err = Console(stderr=True)

app = typer.Typer(
    name="secqa",
    help="Grounded question answering over SEC filings with a reproducible FinanceBench harness.",
    no_args_is_help=True,
    add_completion=False,
    rich_markup_mode=None,
    context_settings={"help_option_names": ["-h", "--help"]},
)
data_app = typer.Typer(help="Fetch the FinanceBench dataset, its PDFs and EDGAR companyfacts.")
ingest_app = typer.Typer(help="Build or extend the DuckDB index (the only write path).")
index_app = typer.Typer(help="Pack, fetch or inspect an index file.")
app.add_typer(data_app, name="data")
app.add_typer(ingest_app, name="ingest")
app.add_typer(index_app, name="index")


# ---------------------------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------------------------


def fail(message: str, code: int = 1) -> None:
    """Print ``error: <message>`` to stderr and exit with ``code``."""
    err.print(f"error: {message}", highlight=False, soft_wrap=True)
    raise typer.Exit(code=code)


def guarded(fn: Callable[..., T]) -> Callable[..., T]:
    """Turn expected failures into a one-line ``error:`` and exit 1 (tracebacks stay for bugs).

    ``SecqaError`` covers configuration, provider, index and guard errors; ``OSError`` and
    ``ValueError`` cover missing files, bad paths and malformed user input from the library.
    """

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> T:
        try:
            return fn(*args, **kwargs)
        except (SecqaError, OSError, ValueError) as exc:
            log.debug("command_failed", error=str(exc), kind=type(exc).__name__)
            fail(str(exc))
        except KeyboardInterrupt:
            fail("interrupted", code=130)
        raise AssertionError("unreachable")  # pragma: no cover - fail() always exits

    return wrapper


def echo_json(payload: Any) -> None:
    """Write JSON to stdout without rich wrapping or highlighting (script-friendly)."""
    typer.echo(json.dumps(payload, indent=2, sort_keys=True, default=str))


def resolve_settings(**overrides: Any) -> Settings:
    """Process settings with non-``None`` command-line overrides applied."""
    updates = {key: value for key, value in overrides.items() if value is not None}
    settings = get_settings()
    return settings.model_copy(update=updates) if updates else settings


@contextmanager
def readonly_store(settings: Settings) -> Iterator[DuckDBStore]:
    """Open ``settings.duckdb_path`` read-only, with a hint when it is missing."""
    path = Path(settings.duckdb_path)
    if not path.is_file():
        raise ConfigError(
            f"index not found at {path}; build it with `secqa ingest financebench` "
            "or download one with `secqa index fetch`"
        )
    store = DuckDBStore(path, read_only=True)
    try:
        yield store
    finally:
        store.close()


@contextmanager
def writable_store(settings: Settings, embedder: Embedder | None) -> Iterator[DuckDBStore]:
    """Open (or create) ``settings.duckdb_path`` for ingest.

    With an ``embedder`` the schema is initialised (idempotent) for its name and width, so a
    second ingest with a different embedder fails with :class:`IndexMismatch` instead of mixing
    vector spaces. Without one the store must already be initialised.
    """
    path = Path(settings.duckdb_path)
    store = DuckDBStore(path, embed_dim=embedder.dim if embedder is not None else None)
    try:
        if embedder is not None:
            store.init_schema(embedder.name, embedder.dim)
        elif store.embedder_name is None:
            store.close()
            raise ConfigError(
                f"{path} is not an initialised index; run `secqa ingest ...` first or pass "
                "--embedder so the schema can be created"
            )
        yield store
    finally:
        store.close()


def build_embedder(settings: Settings, spec: str | None) -> Embedder:
    """``get_embedder`` for ``spec`` (or the configured default), imported lazily.

    Lazy so ``secqa --help`` and ``secqa report`` never import scikit-learn / torch.
    """
    from secqa.embeddings import get_embedder

    return get_embedder(spec or settings.embedder, settings)


def edgar_client(settings: Settings, cache_dir: Path) -> EdgarClient:
    """A rate-limited, cached EDGAR client (``SEC_USER_AGENT`` required)."""
    return EdgarClient.from_settings(settings, cache_dir=cache_dir)


def parse_years(text: str) -> list[int]:
    """``'2022-2024'`` -> ``[2022, 2023, 2024]``; ``'2021,2023'`` -> ``[2021, 2023]``.

    Raises:
        ValueError: on anything that is not years, ranges and commas.
    """
    years: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_text, _, end_text = part.partition("-")
            start, end = int(start_text), int(end_text)
            if end < start:
                raise ValueError(f"year range {part!r} ends before it starts")
            years.extend(range(start, end + 1))
        else:
            years.append(int(part))
    if not years:
        raise ValueError(f"no years in {text!r}; expected e.g. '2022-2024' or '2021,2023'")
    for year in years:
        if not 1993 <= year <= 2100:  # EDGAR electronic filings start in 1993
            raise ValueError(f"year {year} is outside the EDGAR range 1993-2100")
    return sorted(set(years))


def parse_csv(text: str) -> tuple[str, ...]:
    """``'10-K, 10-Q'`` -> ``('10-K', '10-Q')`` (blank entries dropped)."""
    items = tuple(item.strip() for item in text.split(",") if item.strip())
    if not items:
        raise ValueError(f"expected a comma-separated list, got {text!r}")
    return items


def load_questions(path: Path | None, cache_dir: Path, split: str, revision: str | None) -> list:
    """Questions from an explicit JSONL file, else the (cached) FinanceBench split."""
    if path is not None:
        return load_questions_jsonl(path)
    return load_financebench(cache_dir, split, revision)


def print_kv(title: str, rows: Iterable[tuple[str, Any]]) -> None:
    """A two-column key/value table."""
    table = Table(title=title, show_header=False, box=None, pad_edge=False)
    table.add_column(style="bold")
    table.add_column()
    for key, value in rows:
        table.add_row(key, str(value))
    out.print(table)


# ---------------------------------------------------------------------------------------------
# root callback
# ---------------------------------------------------------------------------------------------


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"secqa {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        bool,
        typer.Option(
            "--version", callback=_version_callback, is_eager=True, help="Print the version."
        ),
    ] = False,
    verbose: Annotated[
        bool, typer.Option("--verbose", "-v", help="Debug-level logs on stderr.")
    ] = False,
    quiet: Annotated[
        bool, typer.Option("--quiet", "-q", help="Warnings and errors only on stderr.")
    ] = False,
    log_json: Annotated[
        bool, typer.Option("--log-json", help="One JSON object per log line instead of console.")
    ] = False,
) -> None:
    """Configure logging and resolve settings afresh for whichever command runs next.

    The settings singleton is cleared so an invocation always reflects the current environment
    (a CLI invocation is normally a fresh process; tests and notebooks run several in one).
    """
    level = logging.DEBUG if verbose else logging.WARNING if quiet else logging.INFO
    configure_logging(json=log_json, level=level)
    get_settings.cache_clear()


# ---------------------------------------------------------------------------------------------
# doctor
# ---------------------------------------------------------------------------------------------

Status = Literal["ok", "warn", "fail", "skip"]


@dataclass(frozen=True)
class Check:
    """One line of the doctor report."""

    name: str
    status: Status
    detail: str


def _key_present(settings: Settings, vendor: str) -> bool:
    key = settings.openai_api_key if vendor == "openai" else settings.anthropic_api_key
    return key is not None and bool(key.get_secret_value().strip())


def _vendor_model_check(
    http: httpx.Client, settings: Settings, vendor: str, model: str
) -> tuple[Status, str]:
    """GET the vendor's ``/v1/models/<id>``: 200 -> ok, 404 -> fail, 401 -> fail, else warn."""
    if vendor == "openai":
        key = settings.openai_api_key
        url = OPENAI_MODELS_URL + model
        headers = {"Authorization": f"Bearer {key.get_secret_value() if key else ''}"}
    else:
        key = settings.anthropic_api_key
        url = ANTHROPIC_MODELS_URL + model
        headers = {
            "x-api-key": key.get_secret_value() if key else "",
            "anthropic-version": ANTHROPIC_VERSION_HEADER,
        }
    try:
        response = http.get(url, headers=headers)
    except httpx.HTTPError as exc:
        return "warn", f"could not reach {vendor}: {type(exc).__name__}"
    if response.status_code == 200:
        return "ok", f"{vendor} lists model id {model!r}"
    if response.status_code == 404:
        return "fail", f"{vendor} does not know model id {model!r}; fix models.yaml / configs"
    if response.status_code in (401, 403):
        return "fail", f"{vendor} rejected the API key (HTTP {response.status_code})"
    return "warn", f"{vendor} answered HTTP {response.status_code} for {model!r}"


def run_doctor(
    settings: Settings, *, offline: bool = False, http: httpx.Client | None = None
) -> list[Check]:
    """Every environment check, in report order (pure: no printing, no exit).

    Offline with no keys nothing fails: absent keys are ``skip`` (rows stay pending), an
    unbuilt index and a missing User-Agent are ``warn``. Vendor model-id checks run only when
    the vendor key is set and ``offline`` is False; ``http`` is injectable for tests.
    """
    checks: list[Check] = []
    major, minor = sys.version_info[:2]
    checks.append(
        Check(
            "python",
            "ok" if (major, minor) >= (3, 11) else "fail",
            f"{major}.{minor} ({sys.executable})",
        )
    )
    checks.append(Check("secqa", "ok", f"{__version__} git {resolve_git_sha()[:12]}"))
    checks.append(
        Check(
            "settings",
            "ok",
            f"provider={settings.provider} judge={settings.judge_provider} "
            f"embedder={settings.embedder} index={settings.duckdb_path}",
        )
    )
    try:
        settings.validate_provider_keys()
        checks.append(Check("default provider", "ok", f"{settings.provider} is usable"))
    except ConfigError as exc:
        checks.append(Check("default provider", "fail", str(exc)))

    for vendor in ("openai", "anthropic"):
        env_name = f"{vendor.upper()}_API_KEY"
        if _key_present(settings, vendor):
            checks.append(Check(env_name, "ok", "set"))
        else:
            checks.append(Check(env_name, "skip", f"not set; {vendor} rows stay pending"))

    if settings.sec_user_agent and EMAIL_RE.search(settings.sec_user_agent):
        checks.append(Check("SEC_USER_AGENT", "ok", "declared with a contact email"))
    else:
        checks.append(
            Check("SEC_USER_AGENT", "warn", "not set; EDGAR ingest and PDF fallback disabled")
        )

    prices: PriceTable | None = None
    try:
        prices = PriceTable.load(DEFAULT_MODELS_YAML)
        checks.append(
            Check("price table", "ok", f"{len(prices.models)} models, as_of {prices.as_of}")
        )
    except ConfigError as exc:
        checks.append(Check("price table", "fail", str(exc)))

    checks.extend(_model_id_checks(settings, prices, offline=offline, http=http))

    for module, extra in (("openai", "openai"), ("anthropic", "anthropic")):
        if importlib.util.find_spec(module) is not None:
            checks.append(Check(f"{module} sdk", "ok", "installed"))
        else:
            checks.append(
                Check(f"{module} sdk", "skip", f"not installed (uv sync --extra {extra})")
            )

    checks.append(_embedder_check(settings))
    checks.append(_fts_check())
    checks.append(_index_check(settings))
    if settings.cassette_mode == "replay" and not Path(settings.cassette_dir).is_dir():
        checks.append(
            Check("cassettes", "fail", f"replay mode but {settings.cassette_dir} does not exist")
        )
    else:
        checks.append(Check("cassettes", "ok", f"mode={settings.cassette_mode}"))
    return checks


def _model_id_checks(
    settings: Settings, prices: PriceTable | None, *, offline: bool, http: httpx.Client | None
) -> list[Check]:
    """One check per configured vendor model id (priced models + the configured specs)."""
    wanted: dict[tuple[str, str], None] = {}
    for spec in (settings.provider, settings.judge_provider):
        vendor = provider_vendor(spec)
        model = spec.split(":", 1)[1].strip() if ":" in spec else ""
        if vendor in ("openai", "anthropic") and model:
            wanted[(vendor, model)] = None
    if prices is not None:
        for vendor, model in prices.models:
            if vendor in ("openai", "anthropic"):
                wanted[(vendor, model)] = None
    checks: list[Check] = []
    own_client = http is None and not offline
    client = http or (httpx.Client(timeout=VENDOR_CHECK_TIMEOUT_S) if own_client else None)
    try:
        for vendor, model in sorted(wanted):
            name = f"model {vendor}:{model}"
            if prices is not None and not prices.has(vendor, model):
                checks.append(Check(name, "warn", "configured but not priced in models.yaml"))
                continue
            if not _key_present(settings, vendor):
                checks.append(Check(name, "skip", f"no {vendor.upper()}_API_KEY; not checked"))
            elif offline or client is None:
                checks.append(Check(name, "skip", "offline; vendor endpoint not queried"))
            else:
                status, detail = _vendor_model_check(client, settings, vendor, model)
                checks.append(Check(name, status, detail))
    finally:
        if own_client and client is not None:
            client.close()
    return checks


def _embedder_check(settings: Settings) -> Check:
    kind = settings.embedder.split(":", 1)[0].strip().lower()
    if kind == "hashing":
        return Check("embedder", "ok", f"{settings.embedder} (no downloads)")
    if kind == "local":
        if importlib.util.find_spec("sentence_transformers") is None:
            return Check(
                "embedder", "warn", f"{settings.embedder} needs `uv sync --extra local` (torch)"
            )
        return Check("embedder", "ok", f"{settings.embedder} via sentence-transformers")
    if kind == "openai":
        if _key_present(settings, "openai"):
            return Check("embedder", "ok", f"{settings.embedder} (OPENAI_API_KEY set)")
        return Check("embedder", "fail", f"{settings.embedder} needs OPENAI_API_KEY")
    return Check("embedder", "fail", f"unknown embedder spec {settings.embedder!r}")


def _fts_check() -> Check:
    try:
        with DuckDBStore(":memory:") as store:
            backend = store.bm25_backend
    except Exception as exc:  # duckdb import / extension load failures are environment issues
        return Check("duckdb fts", "fail", f"cannot open DuckDB: {exc}")
    if backend == "duckdb_fts":
        return Check("duckdb fts", "ok", "fts extension loaded (porter stemmer BM25)")
    return Check(
        "duckdb fts", "warn", "fts extension unavailable; python BM25 fallback (no stemming)"
    )


def _index_check(settings: Settings) -> Check:
    path = Path(settings.duckdb_path)
    if not path.is_file():
        return Check("index", "warn", f"{path} not built (secqa ingest ... / secqa index fetch)")
    try:
        with DuckDBStore(path, read_only=True) as store:
            manifest = store.manifest()
    except (SecqaError, KeyError, OSError) as exc:
        return Check("index", "fail", f"{path}: {exc}")
    return Check(
        "index",
        "ok",
        f"{path}: {manifest.n_documents} docs, {manifest.n_pages} pages, "
        f"{manifest.n_chunks} chunks, {manifest.n_facts} facts, embedder {manifest.embedder} "
        f"dim {manifest.dim}, built {manifest.built_at:%Y-%m-%d}",
    )


_STATUS_STYLE = {"ok": "green", "warn": "yellow", "fail": "red", "skip": "dim"}


@app.command()
@guarded
def doctor(
    offline: Annotated[
        bool, typer.Option("--offline", help="Skip the vendor model-id checks (no network).")
    ] = False,
    json_out: Annotated[bool, typer.Option("--json", help="Print the checks as JSON.")] = False,
) -> None:
    """Report what this environment can do: keys, model ids, User-Agent, DuckDB FTS, index.

    Exit status is 1 only when a check fails; missing keys are reported as skipped so the
    pending cells of RESULTS.md are explained, not guessed.
    """
    checks = run_doctor(resolve_settings(), offline=offline)
    failed = sum(1 for check in checks if check.status == "fail")
    if json_out:
        echo_json({"checks": [asdict(check) for check in checks], "failed": failed})
    else:
        table = Table(title="secqa doctor", pad_edge=False)
        table.add_column("check", style="bold")
        table.add_column("status")
        table.add_column("detail", overflow="fold")
        for check in checks:
            table.add_row(
                check.name, f"[{_STATUS_STYLE[check.status]}]{check.status}[/]", check.detail
            )
        out.print(table)
        out.print(f"{len(checks)} checks, {failed} failed", highlight=False)
    if failed:
        raise typer.Exit(code=1)


# ---------------------------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------------------------


@data_app.command("financebench")
@guarded
def data_financebench(
    pdfs: Annotated[
        bool, typer.Option("--pdfs", help="Also download the referenced filing PDFs.")
    ] = False,
    revision: Annotated[
        str | None, typer.Option("--revision", help="Pin a Hugging Face dataset revision.")
    ] = None,
    cache_dir: Annotated[
        Path, typer.Option("--cache-dir", help="Local dataset cache (gitignored).")
    ] = DEFAULT_CACHE_DIR,
    split: Annotated[str, typer.Option("--split")] = DEFAULT_SPLIT,
    edgar_cache_dir: Annotated[
        Path, typer.Option("--edgar-cache-dir", help="On-disk cache for doc_link fallbacks.")
    ] = EDGAR_CACHE_DIR,
) -> None:
    """Download the FinanceBench open set (150 questions; CC-BY-NC-4.0, evaluation only).

    Questions are cached as JSONL under --cache-dir; with --pdfs the ~80 filings come from the
    upstream GitHub repository (magic-byte checked) with the question's doc_link as fallback
    through the EDGAR client when SEC_USER_AGENT is set.
    """
    settings = resolve_settings()
    questions = load_financebench(cache_dir, split, revision)
    doc_names = sorted({q.doc_name for q in questions})
    by_type: dict[str, int] = {}
    for q in questions:
        by_type[q.question_type] = by_type.get(q.question_type, 0) + 1
    print_kv(
        "FinanceBench",
        [
            ("questions", len(questions)),
            ("documents", len(doc_names)),
            ("revision", dataset_revision(cache_dir) or "main"),
            ("by question_type", ", ".join(f"{k}={v}" for k, v in sorted(by_type.items()))),
            ("cache", cache_dir),
        ],
    )
    if not pdfs:
        return
    edgar: EdgarClient | None = None
    if settings.sec_user_agent:
        edgar = edgar_client(settings, edgar_cache_dir)
    else:
        err.print(
            "warning: SEC_USER_AGENT is not set; doc_link fallback for missing PDFs is disabled",
            highlight=False,
        )
    try:
        report = download_pdfs(questions, cache_dir / "pdfs", edgar)
    finally:
        if edgar is not None:
            edgar.close()
    print_kv(
        "PDFs",
        [
            ("downloaded", len(report.downloaded)),
            ("cached", len(report.cached)),
            ("fallback", len(report.fallback)),
            ("missing", len(report.missing)),
            ("upstream commit", report.upstream_commit or "unknown"),
            ("seconds", f"{report.seconds:.1f}"),
        ],
    )
    for doc_name in report.missing:
        err.print(f"missing: {doc_name}: {report.errors.get(doc_name, 'unknown')}", highlight=False)


@data_app.command("companies-xbrl")
@guarded
def data_companies_xbrl(
    companies_path: Annotated[
        Path, typer.Option("--companies", help="Curated company -> ticker -> CIK table.")
    ] = DEFAULT_COMPANIES_PATH,
    db: Annotated[Path | None, typer.Option("--db", help="Index file (default: settings).")] = None,
    embedder: Annotated[
        str | None,
        typer.Option("--embedder", help="Only used to create the schema of a brand-new index."),
    ] = None,
    edgar_cache_dir: Annotated[Path, typer.Option("--edgar-cache-dir")] = EDGAR_CACHE_DIR,
) -> None:
    """Load EDGAR companyfacts for every company in data/companies.yaml into xbrl_facts.

    Recreates the curated `financials` view afterwards. Requires SEC_USER_AGENT.
    """
    settings = resolve_settings(duckdb_path=db)
    companies = load_companies(companies_path)
    edgar = edgar_client(settings, edgar_cache_dir)
    needs_schema = not Path(settings.duckdb_path).is_file()
    resolved = build_embedder(settings, embedder) if needs_schema or embedder else None
    try:
        with writable_store(settings, resolved) as store:
            n_facts = load_xbrl_for_companies(store, edgar, companies)
            counts = store.counts()
    finally:
        edgar.close()
    print_kv(
        "XBRL",
        [
            ("companies", len(companies)),
            ("facts loaded", n_facts),
            ("facts in index", counts["facts"]),
            ("index", settings.duckdb_path),
        ],
    )


# ---------------------------------------------------------------------------------------------
# ingest
# ---------------------------------------------------------------------------------------------


@ingest_app.command("financebench")
@guarded
def ingest_financebench(
    pdf_dir: Annotated[Path, typer.Option("--pdf-dir", help="Directory of <doc_name>.pdf.")] = (
        DEFAULT_PDF_DIR
    ),
    embedder: Annotated[
        str | None,
        typer.Option(
            "--embedder", help="hashing | local | local:<model> | openai (default: settings)."
        ),
    ] = None,
    db: Annotated[Path | None, typer.Option("--db", help="Index file (default: settings).")] = None,
    questions: Annotated[
        Path | None,
        typer.Option("--questions", help="JSONL questions whose doc_names to ingest."),
    ] = None,
    cache_dir: Annotated[Path, typer.Option("--cache-dir")] = DEFAULT_CACHE_DIR,
    doc_names: Annotated[
        list[str] | None, typer.Option("--doc-name", help="Ingest only these documents.")
    ] = None,
    all_pdfs: Annotated[
        bool, typer.Option("--all-pdfs", help="Ingest every PDF in --pdf-dir.")
    ] = False,
    companies_path: Annotated[Path, typer.Option("--companies")] = DEFAULT_COMPANIES_PATH,
    rebuild: Annotated[
        bool, typer.Option("--rebuild", help="Re-embed documents even when unchanged.")
    ] = False,
) -> None:
    """Build the FinanceBench index: PDFs -> pages -> chunks -> embeddings -> DuckDB.

    Documents come from --doc-name, --all-pdfs, --questions, or the cached dataset. Unchanged
    PDFs are skipped unless --rebuild; the manifest (git SHA, input hash, dataset revision) is
    refreshed at the end.
    """
    settings = resolve_settings(duckdb_path=db)
    if not pdf_dir.is_dir():
        raise ConfigError(f"PDF directory not found: {pdf_dir} (secqa data financebench --pdfs)")
    if doc_names:
        names = list(doc_names)
    elif all_pdfs:
        names = sorted(path.stem for path in pdf_dir.glob("*.pdf"))
    else:
        names = list(
            dict.fromkeys(
                q.doc_name for q in load_questions(questions, cache_dir, DEFAULT_SPLIT, None)
            )
        )
    if not names:
        raise ConfigError("nothing to ingest: no document names resolved")
    companies = load_companies(companies_path)
    resolved = build_embedder(settings, embedder)
    with writable_store(settings, resolved) as store:
        report = ingest_financebench_corpus(
            store, resolved, pdf_dir, names, companies, skip_unchanged=not rebuild
        )
        manifest = build_manifest(store, resolve_git_sha(), [pdf_dir], dataset_revision(cache_dir))
    print_kv(
        "Ingest",
        [
            ("requested", len(names)),
            ("ingested", report.documents),
            ("unchanged", len(report.unchanged)),
            ("skipped", len(report.skipped)),
            ("pages", report.pages),
            ("chunks", report.chunks),
            ("seconds", f"{report.seconds:.1f}"),
            ("index", settings.duckdb_path),
            ("embedder", f"{manifest.embedder} dim {manifest.dim}"),
            ("inputs sha256", manifest.inputs_sha256[:16]),
            ("totals", f"{manifest.n_documents} docs / {manifest.n_chunks} chunks"),
        ],
    )
    for doc_name in report.skipped:
        err.print(f"skipped: {doc_name} (missing or unreadable PDF)", highlight=False)


@ingest_app.command("ticker")
@guarded
def ingest_ticker_cmd(
    ticker: Annotated[str, typer.Argument(help="Exchange ticker, e.g. NVDA.")],
    forms: Annotated[
        str, typer.Option("--forms", help="Comma-separated, e.g. 10-K,10-Q.")
    ] = "10-K",
    years: Annotated[
        str, typer.Option("--years", help="Period-of-report years: 2022-2024 or 2021,2023.")
    ] = "2020-2025",
    embedder: Annotated[str | None, typer.Option("--embedder")] = None,
    db: Annotated[Path | None, typer.Option("--db")] = None,
    xbrl: Annotated[
        bool, typer.Option("--xbrl/--no-xbrl", help="Also load the company's XBRL facts.")
    ] = True,
    edgar_cache_dir: Annotated[Path, typer.Option("--edgar-cache-dir")] = EDGAR_CACHE_DIR,
) -> None:
    """Ingest a company's EDGAR 10-K/10-Q primary documents (HTML) for the given years.

    Requires SEC_USER_AGENT ("Name email"); every request is rate-limited and cached. The
    manifest's input hash then covers the EDGAR cache directory.
    """
    from secqa.xbrl import create_financials_view, load_companyfacts

    settings = resolve_settings(duckdb_path=db)
    form_list = parse_csv(forms)
    year_list = parse_years(years)
    edgar = edgar_client(settings, edgar_cache_dir)
    resolved = build_embedder(settings, embedder)
    n_facts = 0
    try:
        with writable_store(settings, resolved) as store:
            present = ingest_ticker(
                store, resolved, edgar, ticker, forms=form_list, years=year_list
            )
            if xbrl:
                cik = edgar.cik_for_ticker(ticker)
                n_facts = load_companyfacts(store, edgar.companyfacts(cik), ticker)
                create_financials_view(store)
            inputs = [edgar_cache_dir] if edgar_cache_dir.is_dir() else []
            manifest = build_manifest(
                store, resolve_git_sha(), inputs, store.manifest().dataset_revision
            )
    finally:
        edgar.close()
    print_kv(
        f"Ingest {ticker.upper()}",
        [
            ("forms", ", ".join(form_list)),
            ("years", f"{year_list[0]}-{year_list[-1]}" if len(year_list) > 1 else year_list[0]),
            ("documents present", len(present)),
            ("doc_names", ", ".join(present) or "none"),
            ("xbrl facts loaded", n_facts if xbrl else "skipped"),
            ("index", settings.duckdb_path),
            ("totals", f"{manifest.n_documents} docs / {manifest.n_chunks} chunks"),
        ],
    )


# ---------------------------------------------------------------------------------------------
# index
# ---------------------------------------------------------------------------------------------


@index_app.command("pack")
@guarded
def index_pack(
    output: Annotated[Path, typer.Option("--out", help="Tarball path (.tar.zst).")] = (
        DEFAULT_INDEX_TARBALL
    ),
    db: Annotated[Path | None, typer.Option("--db")] = None,
) -> None:
    """Pack index.duckdb + manifest.json into a zstd tarball (the GitHub Release asset)."""
    settings = resolve_settings(duckdb_path=db)
    packed = pack_index(Path(settings.duckdb_path), output)
    print_kv(
        "Packed",
        [("index", settings.duckdb_path), ("tarball", packed), ("bytes", packed.stat().st_size)],
    )


@index_app.command("fetch")
@guarded
def index_fetch(
    url: Annotated[
        str | None,
        typer.Argument(help="https://, gs:// or file:// URL (default: SECQA_INDEX_URL)."),
    ] = None,
    dest: Annotated[Path | None, typer.Option("--dest", help="Where to put index.duckdb.")] = None,
    force: Annotated[bool, typer.Option("--force", help="Overwrite an existing file.")] = False,
) -> None:
    """Download a packed (or bare) index and verify it against its manifest."""
    settings = resolve_settings(duckdb_path=dest)
    source = url or settings.index_url
    if not source:
        raise ConfigError("no URL given and SECQA_INDEX_URL is not set")
    target = Path(settings.duckdb_path)
    if target.exists() and not force:
        raise ConfigError(f"{target} already exists; pass --force to replace it")
    fetched = fetch_index(source, target)
    with DuckDBStore(fetched, read_only=True) as store:
        manifest = store.manifest()
    print_kv(
        "Fetched",
        [
            ("url", source),
            ("dest", fetched),
            ("bytes", fetched.stat().st_size),
            ("embedder", f"{manifest.embedder} dim {manifest.dim}"),
            ("documents", manifest.n_documents),
            ("chunks", manifest.n_chunks),
        ],
    )


@index_app.command("manifest")
@guarded
def index_manifest(
    db: Annotated[Path | None, typer.Option("--db")] = None,
    json_out: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Print the index manifest (embedder, git SHA, counts, input hash, dataset revision)."""
    settings = resolve_settings(duckdb_path=db)
    with readonly_store(settings) as store:
        manifest = store.manifest()
        counts = store.counts()
        backend = store.bm25_backend
    if json_out:
        echo_json(
            {
                "path": str(settings.duckdb_path),
                "manifest": manifest.model_dump(mode="json"),
                "counts": counts,
                "bm25_backend": backend,
            }
        )
        return
    rows = [(key, value) for key, value in manifest.model_dump(mode="json").items()]
    rows.extend([("live counts", counts), ("bm25 backend", backend)])
    print_kv(f"Index {settings.duckdb_path}", rows)


# ---------------------------------------------------------------------------------------------
# ask
# ---------------------------------------------------------------------------------------------


def _answer_question(
    settings: Settings,
    question: str,
    *,
    mode: str,
    filters: RetrievalFilters | None,
    k: int,
    strategy: str,
    max_cost_usd: float,
) -> Answer:
    """Run one question through the chosen mode with the same objects the API uses."""
    from secqa.agent import AgentLoop, ToolRuntime
    from secqa.grounding import CitationVerifier
    from secqa.rag import RagPipeline, answer_closed_book
    from secqa.retrieval import Retriever

    provider = get_provider(settings.provider, settings)
    prices = PriceTable.load(DEFAULT_MODELS_YAML)
    if not prices.has(provider.provider, provider.model):
        raise ConfigError(
            f"{provider.provider}:{provider.model} is not priced in models.yaml; "
            "add it before spending money on it"
        )
    verifier = CitationVerifier()
    if mode == "closed_book":
        return answer_closed_book(question, provider, prices)
    with readonly_store(settings) as store:
        embedder = build_embedder(settings, None)
        retriever = Retriever(store, embedder, strategy=strategy, k=k)  # type: ignore[arg-type]  # validated
        if mode == "rag":
            pipeline = RagPipeline(retriever, provider, verifier, prices, k=k)
            return pipeline.answer(question, filters=filters)
        runtime = ToolRuntime(store, retriever, max_result_chars=AGENT_TOOL_RESULT_CHARS)
        loop = AgentLoop(
            provider,
            runtime,
            verifier,
            prices,
            max_steps=settings.max_agent_steps,
            max_cost_usd=max_cost_usd,
            wall_clock_s=float(settings.request_timeout_s),
        )
        return loop.run(question, filters=filters)


def format_value(value: float) -> str:
    """``1577000000.0`` -> ``'1,577,000,000'``; ``0.1234`` -> ``'0.1234'`` (no exponent form)."""
    if value.is_integer() and abs(value) < 1e15:
        return f"{int(value):,}"
    return f"{value:,.4f}".rstrip("0").rstrip(".")


def _print_answer(answer: Answer, *, show_trace: bool) -> None:
    out.print(
        f"[bold]{answer.mode}[/] via {answer.provider}:{answer.model}  "
        f"{answer.latency_ms:.0f} ms  ${answer.cost_usd:.4f}  "
        f"steps={answer.steps} tool_calls={answer.tool_calls} "
        f"terminated_by={answer.terminated_by}",
        highlight=False,
    )
    out.print()
    out.print(answer.text, highlight=False, soft_wrap=True)
    out.print()
    if answer.value is not None:
        out.print(
            f"value: {format_value(answer.value)} {answer.unit or ''}".rstrip(), highlight=False
        )
    if answer.calculation:
        out.print(f"calculation: {answer.calculation}", highlight=False)
    out.print(
        f"abstained: {'yes' if answer.abstained else 'no'}   "
        f"grounded: {'yes' if answer.grounded else 'no'}   "
        f"citations: {sum(c.verified for c in answer.citations)}/{len(answer.citations)} verified",
        highlight=False,
    )
    if answer.citations:
        table = Table(title="Citations", pad_edge=False)
        table.add_column("status")
        table.add_column("source", overflow="fold")
        table.add_column("snippet", overflow="fold")
        for citation in answer.citations:
            if not citation.valid:
                status = "[red]invalid[/]"
            elif citation.verified:
                status = "[green]verified[/]"
            else:
                status = "[yellow]unverified[/]"
            if citation.kind == "chunk":
                source = f"{citation.doc_name} p.{citation.page_num}"
            else:
                source = f"xbrl {citation.tag} FY{citation.fiscal_year} {citation.accn}"
            table.add_row(status, source, citation.snippet)
        out.print(table)
    if show_trace and answer.trace:
        table = Table(title="Trace", pad_edge=False)
        table.add_column("step")
        table.add_column("kind")
        table.add_column("name")
        table.add_column("ms")
        table.add_column("preview / error", overflow="fold")
        for step in answer.trace:
            table.add_row(
                str(step.step),
                step.kind,
                step.name,
                f"{step.latency_ms:.0f}",
                step.error or step.result_preview,
            )
        out.print(table)


@app.command()
@guarded
def ask(
    question: Annotated[str, typer.Argument(help="The question to answer.")],
    ticker: Annotated[str | None, typer.Option("--ticker", help="Restrict to one company.")] = None,
    doc_names: Annotated[
        list[str] | None, typer.Option("--doc-name", help="Restrict to these documents.")
    ] = None,
    fiscal_year: Annotated[int | None, typer.Option("--fiscal-year")] = None,
    form: Annotated[str | None, typer.Option("--form", help="10-K or 10-Q.")] = None,
    mode: Annotated[str, typer.Option("--mode", help="rag | agent | closed_book.")] = "rag",
    provider: Annotated[
        str | None,
        typer.Option(
            "--provider", help="mock | openai:<model> | anthropic:<model> (default: settings)."
        ),
    ] = None,
    k: Annotated[int, typer.Option("--k", min=1, help="Passages retrieved.")] = 8,
    strategy: Annotated[str, typer.Option("--strategy", help="bm25 | dense | hybrid.")] = "hybrid",
    max_cost_usd: Annotated[
        float | None, typer.Option("--max-cost-usd", min=0.0, help="Agent cost cap.")
    ] = None,
    db: Annotated[Path | None, typer.Option("--db")] = None,
    embedder: Annotated[str | None, typer.Option("--embedder")] = None,
    json_out: Annotated[bool, typer.Option("--json", help="Print the Answer as JSON.")] = False,
    show_trace: Annotated[bool, typer.Option("--trace", help="Show every LLM/tool step.")] = False,
) -> None:
    """Answer one question with verified, page-level citations."""
    if mode not in ASK_MODES:
        raise ConfigError(f"mode must be one of {', '.join(ASK_MODES)}; got {mode!r}")
    if strategy not in STRATEGIES:
        raise ConfigError(f"strategy must be one of {', '.join(STRATEGIES)}; got {strategy!r}")
    if not question.strip():
        raise ConfigError("question must not be blank")
    settings = resolve_settings(duckdb_path=db, provider=provider, embedder=embedder)
    if k > settings.max_k:
        raise ConfigError(f"k={k} exceeds SECQA_MAX_K={settings.max_k}")
    filters: RetrievalFilters | None = None
    if ticker or doc_names or fiscal_year is not None or form:
        filters = RetrievalFilters(
            ticker=ticker.upper() if ticker else None,
            doc_names=list(doc_names) if doc_names else None,
            fiscal_year=fiscal_year,
            form=form,
        )
    answer = _answer_question(
        settings,
        question,
        mode=mode,
        filters=filters,
        k=k,
        strategy=strategy,
        max_cost_usd=max_cost_usd if max_cost_usd is not None else settings.max_cost_usd,
    )
    if json_out:
        echo_json(answer.model_dump(mode="json"))
    else:
        _print_answer(answer, show_trace=show_trace)


# ---------------------------------------------------------------------------------------------
# eval / rescore / report
# ---------------------------------------------------------------------------------------------


def ensure_fixture_index(index_path: Path, cfg: EvalConfig, settings: Settings) -> None:
    """Build ``cfg.fixture_pages_path`` into ``index_path`` when it is absent or stale.

    Stale = built with another embedder name or width than ``cfg.embedder`` resolves to; the
    fixture corpus is two documents, so rebuilding is cheaper than explaining a mismatch.
    """
    if cfg.fixture_pages_path is None:
        return
    embedder = build_embedder(settings, cfg.embedder)
    if index_path.is_file():
        with DuckDBStore(index_path, read_only=True) as existing:
            fresh = existing.embedder_name == embedder.name and existing.dim == embedder.dim
        if fresh:
            return
        log.warning("fixture_index_rebuilt", path=str(index_path), reason="embedder changed")
        index_path.unlink()
        Path(f"{index_path}.wal").unlink(missing_ok=True)
    with DuckDBStore(index_path, embed_dim=embedder.dim) as store:
        store.init_schema(embedder.name, embedder.dim)
        build_fixture_index(store, embedder, Path(cfg.fixture_pages_path))
        build_manifest(store, resolve_git_sha(), [Path(cfg.fixture_pages_path)], None)


def _print_summary(run_dir: Path) -> None:
    from secqa.core.contracts import RunSummary
    from secqa.eval.metrics import SUMMARY_NAME

    summary = RunSummary.model_validate(
        json.loads((run_dir / SUMMARY_NAME).read_text(encoding="utf-8"))
    )
    rows: list[tuple[str, Any]] = [
        ("run", run_dir),
        ("config", summary.config_name),
        ("provider", f"{summary.provider}:{summary.model}"),
        ("judge", f"{summary.judge_model} ({summary.judge_version})"),
        ("completed", f"{summary.n_completed}/{summary.n}"),
    ]
    for key in (
        "accuracy",
        "abstain_rate",
        "hallucination_rate",
        "numeric_match_rate",
        "faithfulness",
        "citation_verified_rate",
        "grounded_rate",
        "page_recall_10",
        "overlap_recall_10",
        "gold_page_mrr",
    ):
        value = summary.metrics.get(key)
        text = "n/a" if value is None else f"{value:.3f}"
        ci = summary.ci95.get(key)
        if value is not None and ci is not None:
            text += f"  [{ci[0]:.3f}, {ci[1]:.3f}]"
        rows.append((key, text))
    rows.extend(
        [
            ("latency p50/p95 ms", f"{summary.latency_p50_ms:.0f} / {summary.latency_p95_ms:.0f}"),
            (
                "cost total / per q",
                f"${summary.cost_total_usd:.4f} / ${summary.cost_per_q_usd:.4f}",
            ),
            ("judge cost", f"${summary.judge_cost_usd:.4f}"),
            ("failures", ", ".join(f"{k}={v}" for k, v in sorted(summary.failures.items()))),
        ]
    )
    print_kv("Summary", rows)


@app.command("eval")
@guarded
def eval_cmd(
    config: Annotated[Path, typer.Option("--config", help="configs/<row>.yaml")],
    limit: Annotated[
        int | None, typer.Option("--limit", min=1, help="First N questions only.")
    ] = None,
    resume: Annotated[
        bool, typer.Option("--resume/--no-resume", help="Continue the latest incomplete run.")
    ] = True,
    out_dir: Annotated[Path, typer.Option("--out", help="Results root.")] = DEFAULT_OUT_DIR,
    cassette_dir: Annotated[
        Path | None, typer.Option("--cassette-dir", help="Cassette root (default: settings).")
    ] = None,
    db: Annotated[
        Path | None, typer.Option("--db", help="Index (default: config index_path, then settings).")
    ] = None,
    questions: Annotated[
        Path | None,
        typer.Option("--questions", help="JSONL questions (default: config / dataset)."),
    ] = None,
    cache_dir: Annotated[Path, typer.Option("--cache-dir")] = DEFAULT_CACHE_DIR,
) -> None:
    """Run one row of the evaluation matrix into results/<config>/<run_id>/.

    Runs resume by default (done ids are skipped, cassettes reused). Mock configs build their
    own fixture index when it is missing, so `secqa eval --config configs/rag_mock.yaml` works
    from a clean clone with no keys.
    """
    settings = resolve_settings(duckdb_path=db)
    cfg = load_config(config)
    if limit is not None:
        cfg = cfg.model_copy(update={"limit": limit})
    question_list = load_questions(questions or cfg.questions_path, cache_dir, DEFAULT_SPLIT, None)
    index_path = Path(db) if db is not None else Path(cfg.index_path or settings.duckdb_path)
    ensure_fixture_index(index_path, cfg, settings)
    if not index_path.is_file():
        raise ConfigError(
            f"index not found at {index_path}; build it with `secqa ingest financebench`"
        )
    resolved_cassettes = cassette_dir if cassette_dir is not None else settings.cassette_dir
    with DuckDBStore(index_path, read_only=True) as store:
        run_dir = run_eval(
            cfg,
            question_list,
            store,
            out_dir=out_dir,
            resume=resume,
            cassette_dir=resolved_cassettes,
        )
    _print_summary(run_dir)


@app.command("rescore")
@guarded
def rescore_cmd(
    run: Annotated[Path, typer.Option("--run", help="results/<config>/<run_id>")],
    judge: Annotated[
        str | None, typer.Option("--judge", help="Re-judge with this provider spec.")
    ] = None,
    cassette_dir: Annotated[
        Path | None, typer.Option("--cassette-dir", help="Override the run's cassette path.")
    ] = None,
    db: Annotated[Path | None, typer.Option("--db")] = None,
    questions: Annotated[Path | None, typer.Option("--questions")] = None,
    cache_dir: Annotated[Path, typer.Option("--cache-dir")] = DEFAULT_CACHE_DIR,
) -> None:
    """Recompute a run's predictions, metrics and summary from its cassettes (no keys needed).

    With --judge the run is re-judged by that model (new cassettes recorded in place); the
    previous predictions are kept as predictions.previous.jsonl.
    """
    settings = resolve_settings(duckdb_path=db)
    raw = read_run_config(run)
    cfg = EvalConfig.model_validate(raw["config"])
    question_list = load_questions(questions or cfg.questions_path, cache_dir, DEFAULT_SPLIT, None)
    judge_provider = get_provider(judge, settings) if judge else None
    store: DuckDBStore | None = None
    if db is not None:
        store = DuckDBStore(Path(db), read_only=True)
    try:
        rescore(run, question_list, judge_provider, store=store, cassette_dir=cassette_dir)
    finally:
        if store is not None:
            store.close()
    _print_summary(run)


@app.command()
@guarded
def report(
    results_dir: Annotated[Path, typer.Argument(help="Results root.")] = DEFAULT_OUT_DIR,
    output: Annotated[
        Path, typer.Option("--out", help="Markdown file to write ('-' for stdout).")
    ] = DEFAULT_RESULTS_MD,
    configs_dir: Annotated[Path, typer.Option("--configs")] = DEFAULT_CONFIGS_DIR,
) -> None:
    """Regenerate RESULTS.md from committed summary.json files (pending rows stay pending)."""
    text = render_results_md(results_dir, configs_dir)
    if str(output) == "-":
        typer.echo(text, nl=False)
        return
    output.write_text(text, encoding="utf-8")
    n_pending = text.count("pending (not run)")
    n_complete = sum(1 for line in text.splitlines() if line.endswith("| complete |"))
    print_kv(
        "Report",
        [("written", output), ("complete rows", n_complete), ("pending rows", n_pending)],
    )


# ---------------------------------------------------------------------------------------------
# serve / export
# ---------------------------------------------------------------------------------------------


@app.command()
@guarded
def serve(
    host: Annotated[str, typer.Option("--host")] = "0.0.0.0",
    port: Annotated[int, typer.Option("--port", min=1, max=65535)] = 8080,
    reload: Annotated[bool, typer.Option("--reload", help="Auto-reload on code changes.")] = False,
    workers: Annotated[int, typer.Option("--workers", min=1)] = 1,
    db: Annotated[Path | None, typer.Option("--db", help="Index file (exported as env).")] = None,
) -> None:
    """Run the FastAPI service with uvicorn (same factory the container uses)."""
    import uvicorn

    if db is not None:
        # The app factory runs in each worker process and reads settings from the environment.
        os.environ["SECQA_DUCKDB_PATH"] = str(db)
        get_settings.cache_clear()
    settings = resolve_settings()
    settings.validate_provider_keys()  # fail here, not inside a worker
    out.print(
        f"serving secqa {__version__} on http://{host}:{port} "
        f"(provider={settings.provider}, index={settings.duckdb_path}, docs at /docs)",
        highlight=False,
    )
    uvicorn.run(
        "secqa.api.app:create_app",
        factory=True,
        host=host,
        port=port,
        reload=reload,
        workers=workers,
        log_level="info",
    )


@app.command()
@guarded
def export(
    output: Annotated[Path, typer.Option("--out", help="Directory for <table>.parquet.")] = (
        DEFAULT_PARQUET_DIR
    ),
    db: Annotated[Path | None, typer.Option("--db")] = None,
) -> None:
    """Export every index table to Parquet (DuckDB-independent portability)."""
    settings = resolve_settings(duckdb_path=db)
    with readonly_store(settings) as store:
        store.export_parquet(output)
    files = sorted(path.name for path in output.glob("*.parquet"))
    print_kv("Exported", [("directory", output), ("files", ", ".join(files))])


__all__ = [
    "AGENT_TOOL_RESULT_CHARS",
    "ASK_MODES",
    "Check",
    "app",
    "ensure_fixture_index",
    "format_value",
    "guarded",
    "main",
    "parse_csv",
    "parse_years",
    "resolve_settings",
    "run_doctor",
]


if __name__ == "__main__":  # pragma: no cover - `python -m secqa` goes through __main__.py
    app()
