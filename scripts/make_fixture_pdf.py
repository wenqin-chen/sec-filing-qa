#!/usr/bin/env python
"""Render the synthetic fixture corpus as PDFs (one file per document, one page per entry).

Input is ``tests/fixtures/eval_fixture_pages.json`` (``{doc_name: {..., pages: [text, ...]}}``),
the same hand-written corpus the mock eval and the demo index use. The PDFs let you exercise the
real ingestion path (``secqa ingest financebench --pdf-dir <out>``) and the page-indexing check
without any third-party document. Text is drawn as plain wrapped lines so pypdfium2 extracts it
verbatim. Output goes under ``data/raw/`` (gitignored); PDFs are never committed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from secqa.core.errors import ConfigError
from secqa.core.logging import get_logger

log = get_logger("secqa.make_fixture_pdf")

DEFAULT_PAGES_JSON = Path("tests/fixtures/eval_fixture_pages.json")
DEFAULT_OUT_DIR = Path("data/raw/fixture_pdfs")
LINE_WIDTH = 90
FONT = ("Helvetica", 11)
MARGIN = 72
LINE_HEIGHT = 14


def wrap(text: str, width: int = LINE_WIDTH) -> list[str]:
    """Greedy word wrap (no hyphenation) so extracted text equals the input modulo whitespace."""
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


def write_pdf(pages: list[str], out: Path) -> Path:
    """Write ``pages`` (one string per physical page) to ``out`` with reportlab."""
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen.canvas import Canvas
    except ImportError as exc:  # reportlab is a dev extra
        raise ConfigError("reportlab is required: uv sync --extra dev") from exc
    if not pages:
        raise ValueError("a PDF needs at least one page")
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    width, height = letter
    canvas = Canvas(str(out), pagesize=letter)
    for text in pages:
        canvas.setFont(*FONT)
        y = height - MARGIN
        for line in wrap(text):
            canvas.drawString(MARGIN, y, line)
            y -= LINE_HEIGHT
        canvas.showPage()
    canvas.save()
    return out


def load_pages_json(path: Path) -> dict[str, list[str]]:
    """``{doc_name: [page text, ...]}`` from the fixture corpus file."""
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigError(f"cannot read {path}: {exc}") from exc
    if not isinstance(raw, dict) or not raw:
        raise ConfigError(f"{path} must map doc_name -> document")
    out: dict[str, list[str]] = {}
    for doc_name, doc in raw.items():
        pages = doc.get("pages") if isinstance(doc, dict) else None
        if not isinstance(pages, list) or not pages:
            raise ConfigError(f"{path}: {doc_name} needs a non-empty 'pages' list")
        out[str(doc_name)] = [str(p) for p in pages]
    return out


def render_all(pages_json: Path, out_dir: Path, only: list[str] | None = None) -> list[Path]:
    """Write one PDF per document (optionally restricted to ``only``); returns the paths."""
    corpus = load_pages_json(pages_json)
    wanted = set(only) if only else set(corpus)
    unknown = sorted(wanted - set(corpus))
    if unknown:
        raise ConfigError(f"unknown documents {unknown}; available: {sorted(corpus)}")
    written: list[Path] = []
    for doc_name in sorted(wanted):
        path = write_pdf(corpus[doc_name], Path(out_dir) / f"{doc_name}.pdf")
        written.append(path)
        log.info(
            "fixture_pdf_written",
            doc_name=doc_name,
            path=str(path),
            n_pages=len(corpus[doc_name]),
        )
    return written


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--pages-json", type=Path, default=DEFAULT_PAGES_JSON)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--doc", action="append", default=None, help="restrict to this doc_name")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Entry point; 0 on success, 2 on a configuration error."""
    args = parse_args(argv)
    try:
        written = render_all(args.pages_json, args.out_dir, args.doc)
    except ConfigError as exc:
        print(f"[config] {exc}", file=sys.stderr)
        return 2
    for path in written:
        print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
