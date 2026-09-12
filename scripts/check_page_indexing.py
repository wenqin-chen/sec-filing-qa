#!/usr/bin/env python
"""Gate for the FinanceBench page-offset assumption (SPEC 4.1).

FinanceBench's ``evidence_page_num`` is 0-indexed; secqa stores 1-based physical pages and the
loader adds one. This script samples questions (seed 0), extracts their PDFs with the same
``pypdfium2`` path the index uses, and asserts that each evidence text is found on its 1-based
gold page after whitespace/punctuation normalisation. For every miss it reports on which
neighbouring page (offset -2..+2) the text *was* found, so an off-by-one shows up as a spike at
``+1`` or ``-1`` rather than as a vague recall drop.

No recall number is published until the pass rate is >= ``--min-pass`` (default 90%). The report
block is written between ``<!-- page-indexing:start -->`` / ``<!-- page-indexing:end -->``
markers in ``--out`` (default ``docs/DATA.md``; created when absent, other content preserved).
No dataset text is written: the report carries ids, pages and counts only.
"""

from __future__ import annotations

import argparse
import random
import re
import sys
import unicodedata
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from secqa.core.contracts import FBQuestion
from secqa.core.errors import ConfigError
from secqa.core.logging import get_logger
from secqa.eval.financebench import DEFAULT_CACHE_DIR, load_financebench
from secqa.ingest import extract_pdf_pages

log = get_logger("secqa.check_page_indexing")

DEFAULT_PDF_DIR = DEFAULT_CACHE_DIR / "pdfs"
DEFAULT_OUT = Path("docs/DATA.md")
DEFAULT_N = 25
DEFAULT_SEED = 0
DEFAULT_MIN_PASS = 0.9
DEFAULT_WINDOW = 2
MIN_EVIDENCE_CHARS = 20
START_MARKER = "<!-- page-indexing:start -->"
END_MARKER = "<!-- page-indexing:end -->"

_NON_ALNUM_RE = re.compile(r"[^0-9a-z]+")


def normalise(text: str) -> str:
    """NFKC, casefold, and collapse every non-alphanumeric run to one space."""
    folded = unicodedata.normalize("NFKC", text).casefold()
    return _NON_ALNUM_RE.sub(" ", folded).strip()


@dataclass(frozen=True)
class EvidenceHit:
    """Where one evidence text was found relative to its gold page."""

    doc_name: str
    gold_page: int
    found_offsets: tuple[int, ...]  # offsets (page - gold_page) where the text occurs
    skipped: bool = False  # evidence too short to be a meaningful substring test

    @property
    def on_gold(self) -> bool:
        return 0 in self.found_offsets


@dataclass(frozen=True)
class QuestionResult:
    """Outcome for one question: passes when every checkable evidence is on its gold page."""

    id: str
    doc_name: str
    hits: tuple[EvidenceHit, ...]
    error: str | None = None

    @property
    def checkable(self) -> bool:
        return self.error is None and any(not h.skipped for h in self.hits)

    @property
    def passed(self) -> bool:
        return self.checkable and all(h.on_gold for h in self.hits if not h.skipped)


@dataclass
class Report:
    """Aggregate over the sample."""

    results: list[QuestionResult] = field(default_factory=list)
    min_pass: float = DEFAULT_MIN_PASS

    @property
    def checkable(self) -> list[QuestionResult]:
        return [r for r in self.results if r.checkable]

    @property
    def n_passed(self) -> int:
        return sum(1 for r in self.checkable if r.passed)

    @property
    def pass_rate(self) -> float | None:
        checkable = self.checkable
        return self.n_passed / len(checkable) if checkable else None

    @property
    def ok(self) -> bool:
        rate = self.pass_rate
        return rate is not None and rate >= self.min_pass

    def offset_histogram(self) -> Counter[str]:
        """Where missed evidence was found: ``'+1'``, ``'-1'`` ... or ``'not found'``."""
        counts: Counter[str] = Counter()
        for result in self.checkable:
            for hit in result.hits:
                if hit.skipped or hit.on_gold:
                    continue
                others = [o for o in hit.found_offsets if o != 0]
                if not others:
                    counts["not found"] += 1
                for offset in others:
                    counts[f"{offset:+d}"] += 1
        return counts


# ---------------------------------------------------------------------------------------------
# core checks (pure; tested on a reportlab PDF)
# ---------------------------------------------------------------------------------------------


def locate_evidence(
    evidence_text: str, pages: dict[int, str], gold_page: int, window: int = DEFAULT_WINDOW
) -> tuple[int, ...]:
    """Offsets within ``[-window, +window]`` at which ``evidence_text`` occurs in ``pages``.

    ``pages`` maps 1-based page numbers to *raw* page text; both sides are normalised here.
    """
    needle = normalise(evidence_text)
    if not needle:
        return ()
    found: list[int] = []
    for offset in range(-window, window + 1):
        text = pages.get(gold_page + offset)
        if text is not None and needle in normalise(text):
            found.append(offset)
    return tuple(found)


def check_question(
    q: FBQuestion, pages_by_doc: dict[str, dict[int, str]], window: int = DEFAULT_WINDOW
) -> QuestionResult:
    """Locate every evidence entry of ``q`` in the extracted pages of its documents."""
    hits: list[EvidenceHit] = []
    for evidence in q.evidence:
        pages = pages_by_doc.get(evidence.doc_name)
        if pages is None:
            return QuestionResult(q.id, q.doc_name, (), error=f"no pages for {evidence.doc_name}")
        if len(normalise(evidence.text)) < MIN_EVIDENCE_CHARS:
            hits.append(EvidenceHit(evidence.doc_name, evidence.page_num, (), skipped=True))
            continue
        offsets = locate_evidence(evidence.text, pages, evidence.page_num, window)
        hits.append(EvidenceHit(evidence.doc_name, evidence.page_num, offsets))
    if not hits:
        return QuestionResult(q.id, q.doc_name, (), error="question has no evidence")
    return QuestionResult(q.id, q.doc_name, tuple(hits))


def sample_questions(
    questions: Sequence[FBQuestion], n: int, seed: int, available_docs: set[str]
) -> list[FBQuestion]:
    """Deterministic sample of ``n`` questions whose documents are all available locally."""
    eligible = [
        q for q in questions if q.evidence and all(e.doc_name in available_docs for e in q.evidence)
    ]
    eligible.sort(key=lambda q: q.id)
    if n >= len(eligible):
        return eligible
    return sorted(random.Random(seed).sample(eligible, n), key=lambda q: q.id)


def load_pages(pdf_dir: Path, doc_names: set[str]) -> dict[str, dict[int, str]]:
    """Extract every requested PDF once: ``{doc_name: {page_num: text}}``."""
    out: dict[str, dict[int, str]] = {}
    for doc_name in sorted(doc_names):
        path = pdf_dir / f"{doc_name}.pdf"
        try:
            pages = extract_pdf_pages(path, doc_name)
        except (FileNotFoundError, ValueError) as exc:
            log.error("pdf_unreadable", doc_name=doc_name, error=str(exc))
            continue
        out[doc_name] = {p.page_num: p.text for p in pages}
    return out


def run(
    questions: Sequence[FBQuestion],
    pdf_dir: Path,
    *,
    n: int = DEFAULT_N,
    seed: int = DEFAULT_SEED,
    window: int = DEFAULT_WINDOW,
    min_pass: float = DEFAULT_MIN_PASS,
) -> Report:
    """Sample, extract, check; returns the :class:`Report`."""
    available = {p.stem for p in Path(pdf_dir).glob("*.pdf")}
    sample = sample_questions(questions, n, seed, available)
    docs = {e.doc_name for q in sample for e in q.evidence}
    pages_by_doc = load_pages(Path(pdf_dir), docs)
    report = Report(min_pass=min_pass)
    for q in sample:
        report.results.append(check_question(q, pages_by_doc, window))
    log.info(
        "page_indexing_checked",
        n_sampled=len(sample),
        n_checkable=len(report.checkable),
        n_passed=report.n_passed,
        pass_rate=report.pass_rate,
        offsets=dict(report.offset_histogram()),
    )
    return report


# ---------------------------------------------------------------------------------------------
# markdown report
# ---------------------------------------------------------------------------------------------


def render_report(report: Report, *, now: datetime | None = None, seed: int = DEFAULT_SEED) -> str:
    """Markdown block (ids, pages and counts only -- never dataset text)."""
    stamp = (now or datetime.now(UTC)).isoformat(timespec="seconds")
    rate = report.pass_rate
    verdict = "PASS" if report.ok else "FAIL"
    lines = [
        START_MARKER,
        "### Page-indexing check",
        "",
        f"Generated by `scripts/check_page_indexing.py` on {stamp} (seed {seed}).",
        "",
        "Assumption under test: FinanceBench `evidence_page_num` is 0-indexed, so secqa uses "
        "`page_num = evidence_page_num + 1` (1-based physical pages from pypdfium2).",
        "",
        f"- Questions sampled: {len(report.results)}",
        f"- Checkable (PDF readable, evidence >= {MIN_EVIDENCE_CHARS} chars): "
        f"{len(report.checkable)}",
        f"- Evidence found on the gold page: {report.n_passed}/{len(report.checkable)}"
        + (f" ({rate:.1%})" if rate is not None else ""),
        f"- Threshold: {report.min_pass:.0%} -> **{verdict}**",
        "",
    ]
    histogram = report.offset_histogram()
    if histogram:
        lines.append("Where missed evidence was found (page offset from gold):")
        lines.append("")
        lines.append("| offset | count |")
        lines.append("|---|---|")
        for key, count in sorted(histogram.items()):
            lines.append(f"| {key} | {count} |")
        lines.append("")
    failures = [r for r in report.checkable if not r.passed]
    errors = [r for r in report.results if r.error]
    if failures:
        lines.append("Failed questions (id, doc, gold pages):")
        lines.append("")
        for r in failures:
            pages = ", ".join(str(h.gold_page) for h in r.hits if not h.skipped)
            lines.append(f"- `{r.id}` {r.doc_name} p.{pages}")
        lines.append("")
    if errors:
        lines.append("Skipped (not checkable):")
        lines.append("")
        for r in errors:
            lines.append(f"- `{r.id}` {r.doc_name}: {r.error}")
        lines.append("")
    lines.append(END_MARKER)
    return "\n".join(lines) + "\n"


def update_markdown(path: Path, block: str) -> Path:
    """Write ``block`` between the markers in ``path``; append or create when absent."""
    path = Path(path)
    if path.is_file():
        existing = path.read_text(encoding="utf-8")
        start = existing.find(START_MARKER)
        end = existing.find(END_MARKER)
        if start != -1 and end != -1 and end > start:
            end += len(END_MARKER)
            tail = existing[end:]
            if tail.startswith("\n"):
                tail = tail[1:]
            content = existing[:start] + block + tail
        else:
            sep = "" if existing.endswith("\n") or not existing else "\n"
            content = existing + sep + "\n" + block
    else:
        content = "# Data notes\n\n" + block
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--pdf-dir", type=Path, default=DEFAULT_PDF_DIR)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--n", type=int, default=DEFAULT_N)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--window", type=int, default=DEFAULT_WINDOW)
    parser.add_argument("--min-pass", type=float, default=DEFAULT_MIN_PASS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--no-write", action="store_true", help="print the report only")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Entry point; 0 when the pass rate meets the threshold, 1 otherwise, 2 on config error."""
    args = parse_args(argv)
    if args.n < 1 or not 0.0 <= args.min_pass <= 1.0 or args.window < 0:
        print("[config] --n >= 1, 0 <= --min-pass <= 1, --window >= 0", file=sys.stderr)
        return 2
    try:
        questions = load_financebench(cache_dir=args.cache_dir)
    except ConfigError as exc:
        print(f"[config] {exc}", file=sys.stderr)
        return 2
    report = run(
        questions,
        args.pdf_dir,
        n=args.n,
        seed=args.seed,
        window=args.window,
        min_pass=args.min_pass,
    )
    block = render_report(report, seed=args.seed)
    print(block)
    if not args.no_write:
        update_markdown(args.out, block)
    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main())
