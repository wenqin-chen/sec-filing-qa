"""Shared pytest fixtures.

Everything here works offline: no API keys, no network, no model downloads. Module-specific
fixtures live in ``tests/<module>/conftest.py``.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest

from secqa.core.settings import Settings, get_settings

# Environment variables that must never leak from the developer's shell into the test process.
_SCRUBBED_ENV = (
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "SEC_USER_AGENT",
    "SECQA_API_KEY",
    "SECQA_PROVIDER",
    "SECQA_EMBEDDER",
    "SECQA_DUCKDB_PATH",
    "SECQA_INDEX_URL",
    "SECQA_CASSETTE_MODE",
    "SECQA_CASSETTE_DIR",
    "SECQA_LOG_JSON",
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Scrub secqa-related env vars and the settings cache so every test starts from defaults."""
    for name in _SCRUBBED_ENV:
        monkeypatch.delenv(name, raising=False)
    for name in list(os.environ):
        if name.upper().startswith("SECQA_"):
            monkeypatch.delenv(name, raising=False)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def tmp_duckdb_path(tmp_path: Path) -> Path:
    """Path to a not-yet-created DuckDB file inside the test's temporary directory."""
    return tmp_path / "index.duckdb"


@pytest.fixture
def settings_override(
    monkeypatch: pytest.MonkeyPatch, tmp_duckdb_path: Path
) -> Callable[..., Settings]:
    """Factory returning a ``Settings`` built from keyword overrides (no ``.env`` file read).

    ``settings_override(provider="mock", max_k=5)`` also exports the corresponding env vars so
    code paths that call ``get_settings()`` see the same values.
    """

    def _make(**overrides: Any) -> Settings:
        values: dict[str, Any] = {"duckdb_path": tmp_duckdb_path, **overrides}
        for key, value in values.items():
            if value is None:
                continue
            env_name = {
                "openai_api_key": "OPENAI_API_KEY",
                "anthropic_api_key": "ANTHROPIC_API_KEY",
                "sec_user_agent": "SEC_USER_AGENT",
            }.get(key, f"SECQA_{key.upper()}")
            monkeypatch.setenv(env_name, str(value))
        get_settings.cache_clear()
        return Settings(_env_file=None, **values)

    return _make


@pytest.fixture
def fixture_pdf_factory(tmp_path: Path) -> Callable[..., Path]:
    """Generate a small synthetic multi-page PDF with reportlab.

    ``fixture_pdf_factory(pages=["page one text", "page two text"], name="FIXTURE_2023_10K")``
    returns the path. Text is drawn as plain lines so pypdfium2 extracts it verbatim.
    """

    def _make(pages: list[str] | None = None, name: str = "FIXTURE_2023_10K") -> Path:
        reportlab_pdfgen = pytest.importorskip("reportlab.pdfgen.canvas")
        pagesizes = pytest.importorskip("reportlab.lib.pagesizes")
        if pages is None:
            pages = [
                "FIXTURE CORP ANNUAL REPORT FISCAL 2023. Item 7. Management Discussion. "
                "Total net sales were $1,577 million in fiscal 2023, an increase of 12% over 2022.",
                "Item 8. Financial Statements. Operating income was $245 million and net income "
                "was $190 million. Capital expenditures totalled $80 million.",
                "Item 9A. Controls and Procedures. Cash and cash equivalents were $410 million "
                "at year end. Long-term debt was $1,200 million.",
            ]
        out = tmp_path / f"{name}.pdf"
        width, height = pagesizes.letter
        canvas = reportlab_pdfgen.Canvas(str(out), pagesize=pagesizes.letter)
        for page_text in pages:
            canvas.setFont("Helvetica", 11)
            y = height - 72
            for line in _wrap(page_text, 90):
                canvas.drawString(72, y, line)
                y -= 14
            canvas.showPage()
        canvas.save()
        return out

    return _make


@pytest.fixture
def fixture_pdf(fixture_pdf_factory: Callable[..., Path]) -> Path:
    """A 3-page synthetic 10-K-like PDF (see ``fixture_pdf_factory`` for the text)."""
    return fixture_pdf_factory()


@pytest.fixture
def respx_router() -> Iterator[Any]:
    """An active ``respx`` router that fails on any un-mocked HTTP call."""
    respx = pytest.importorskip("respx")
    with respx.mock(assert_all_called=False, assert_all_mocked=True) as router:
        yield router


def _wrap(text: str, width: int) -> list[str]:
    """Greedy word wrap used only for drawing fixture PDFs."""
    words = text.split()
    lines: list[str] = []
    current: list[str] = []
    length = 0
    for word in words:
        if current and length + 1 + len(word) > width:
            lines.append(" ".join(current))
            current, length = [word], len(word)
        else:
            current.append(word)
            length += len(word) + (1 if length else 0)
    if current:
        lines.append(" ".join(current))
    return lines
