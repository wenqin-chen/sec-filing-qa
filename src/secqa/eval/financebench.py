"""FinanceBench loading: Hugging Face rows -> :class:`~secqa.core.contracts.FBQuestion`.

Two facts about the dataset drive this module (SPEC 4.1):

* ``evidence_page_num`` in the PatronusAI release is **0-indexed** while secqa uses 1-based
  physical page numbers everywhere, so ``page_num = evidence_page_num + 1`` happens exactly here
  (:func:`question_from_row`) and nowhere else.
* The licence is CC-BY-NC-4.0: the dataset is used for evaluation only and never redistributed.
  Rows are cached locally under ``data/raw/financebench/`` (gitignored) as one JSON line per
  question so a re-score can join predictions (which carry only ``financebench_id``) back to the
  text without touching the network. Nothing in this module ever writes dataset text under
  ``results/`` or ``tests/``.

The ``datasets`` import is lazy: the offline test suite never needs it, and the loader is only
exercised for real by a ``@pytest.mark.live`` test.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from secqa.core.contracts import Evidence, FBQuestion
from secqa.core.errors import ConfigError
from secqa.core.logging import get_logger
from secqa.edgar import EdgarClient
from secqa.indexing import PdfDownloadReport, download_financebench_pdfs

log = get_logger(__name__)

DATASET_NAME = "PatronusAI/financebench"
DEFAULT_SPLIT = "train"
DEFAULT_CACHE_DIR = Path("data/raw/financebench")
DATASET_INFO_NAME = "DATASET.json"

# Evidence page keys, in order of preference. The PatronusAI README documents
# ``evidence_page_num``; some mirrors expose the same 0-indexed value as ``page_number``.
_EVIDENCE_PAGE_KEYS: tuple[str, ...] = ("evidence_page_num", "page_number")

DownloadReport = PdfDownloadReport
"""Outcome of :func:`download_pdfs` (the indexing module's report, re-exported by name)."""


# ---------------------------------------------------------------------------------------------
# row conversion
# ---------------------------------------------------------------------------------------------


def evidence_from_row(item: dict[str, Any], default_doc_name: str) -> Evidence:
    """One evidence entry: 0-indexed ``evidence_page_num`` -> 1-based ``page_num``.

    Raises:
        ValueError: when no page key is present or the page index is negative.
    """
    raw_page: Any = None
    for key in _EVIDENCE_PAGE_KEYS:
        if item.get(key) is not None:
            raw_page = item[key]
            break
    if raw_page is None:
        raise ValueError(f"evidence entry has no page number (keys {list(item)})")
    try:
        page_index = int(raw_page)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"evidence page {raw_page!r} is not an integer") from exc
    if page_index < 0:
        raise ValueError(f"evidence page index must be >= 0 (0-indexed), got {page_index}")
    text = str(item.get("evidence_text") or "")
    return Evidence(
        doc_name=str(item.get("doc_name") or default_doc_name),
        page_num=page_index + 1,
        text=text,
    )


def question_from_row(row: dict[str, Any]) -> FBQuestion:
    """Convert one Hugging Face row to an :class:`FBQuestion` (page numbers made 1-based).

    Raises:
        ValueError: when a required field (``financebench_id``, ``doc_name``, ``question``,
            ``answer``) is missing, or an evidence entry has no usable page number.
    """
    missing = [
        key for key in ("financebench_id", "doc_name", "question", "answer") if not row.get(key)
    ]
    if missing:
        raise ValueError(f"FinanceBench row is missing {missing}")
    doc_name = str(row["doc_name"])
    raw_evidence = row.get("evidence") or []
    if isinstance(raw_evidence, dict):  # a single entry rather than a list
        raw_evidence = [raw_evidence]
    evidence = [
        evidence_from_row(item, doc_name) for item in raw_evidence if isinstance(item, dict)
    ]
    doc_period = row.get("doc_period")
    return FBQuestion(
        id=str(row["financebench_id"]),
        company=str(row.get("company") or ""),
        doc_name=doc_name,
        question_type=str(row.get("question_type") or "unknown"),
        question=str(row["question"]),
        answer=str(row["answer"]),
        justification=str(row.get("justification") or ""),
        doc_link=str(row["doc_link"]) if row.get("doc_link") else None,
        doc_period=str(doc_period) if doc_period is not None else None,
        evidence=evidence,
    )


def questions_from_rows(rows: Iterable[dict[str, Any]]) -> list[FBQuestion]:
    """Convert every row, rejecting duplicate ids (a duplicate would double-count a question)."""
    questions: list[FBQuestion] = []
    seen: set[str] = set()
    for row in rows:
        question = question_from_row(row)
        if question.id in seen:
            raise ValueError(f"duplicate financebench_id {question.id!r}")
        seen.add(question.id)
        questions.append(question)
    return questions


# ---------------------------------------------------------------------------------------------
# local JSONL cache (our own FBQuestion shape, page numbers already 1-based)
# ---------------------------------------------------------------------------------------------


def save_questions_jsonl(questions: Iterable[FBQuestion], path: Path) -> int:
    """Write questions as one JSON object per line; returns the number written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as fh:
        for question in questions:
            fh.write(json.dumps(question.model_dump(mode="json"), ensure_ascii=False) + "\n")
            n += 1
    return n


def load_questions_jsonl(path: Path) -> list[FBQuestion]:
    """Read questions written by :func:`save_questions_jsonl` (or a hand-written fixture).

    Raises:
        ConfigError: when the file is missing, a line is not valid JSON, or an id repeats.
    """
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"questions file not found: {path}")
    questions: list[FBQuestion] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            try:
                question = FBQuestion.model_validate(json.loads(line))
            except (ValueError, TypeError) as exc:
                raise ConfigError(f"{path}:{line_no}: invalid question record: {exc}") from exc
            if question.id in seen:
                raise ConfigError(f"{path}:{line_no}: duplicate financebench_id {question.id!r}")
            seen.add(question.id)
            questions.append(question)
    log.info("questions_loaded", path=str(path), n=len(questions))
    return questions


def cached_questions_path(cache_dir: Path, split: str = DEFAULT_SPLIT) -> Path:
    """``<cache_dir>/financebench_<split>.jsonl``."""
    return Path(cache_dir) / f"financebench_{split}.jsonl"


# ---------------------------------------------------------------------------------------------
# Hugging Face
# ---------------------------------------------------------------------------------------------


def _load_hf_rows(
    split: str, revision: str | None, cache_dir: Path
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Fetch the rows with ``datasets`` (lazy import); returns ``(rows, dataset_info)``.

    Separated so tests can replace it with an in-memory stub; only this function touches the
    network.
    """
    try:
        import datasets  # noqa: PLC0415 - lazy: optional at test time, slow to import
    except ImportError as exc:  # pragma: no cover - datasets is a declared dependency
        raise ConfigError("the 'datasets' package is required to load FinanceBench") from exc
    dataset = datasets.load_dataset(
        DATASET_NAME, split=split, revision=revision, cache_dir=str(cache_dir / "hf")
    )
    rows = [dict(row) for row in dataset]
    fingerprint = getattr(dataset, "_fingerprint", None)
    info = {
        "dataset": DATASET_NAME,
        "split": split,
        "revision": revision or "main",
        "fingerprint": str(fingerprint) if fingerprint else None,
        "n_rows": len(rows),
        "columns": list(dataset.column_names),
    }
    return rows, info


def load_financebench(
    cache_dir: Path = DEFAULT_CACHE_DIR,
    split: str = DEFAULT_SPLIT,
    revision: str | None = None,
) -> list[FBQuestion]:
    """Load the FinanceBench open set as :class:`FBQuestion` records (page numbers 1-based).

    The first call downloads the split with ``datasets.load_dataset`` and caches the converted
    questions as ``<cache_dir>/financebench_<split>.jsonl`` next to a ``DATASET.json`` that
    records the split, revision, fingerprint and row count; later calls read the JSONL and never
    touch the network. Pass ``revision`` to pin a dataset commit (recorded in ``DATASET.json``);
    a different revision than the cached one forces a re-download.

    Raises:
        ConfigError: when the download fails or the cached files are unreadable.
    """
    cache_dir = Path(cache_dir)
    questions_path = cached_questions_path(cache_dir, split)
    info_path = cache_dir / DATASET_INFO_NAME
    if questions_path.is_file() and info_path.is_file():
        try:
            info = json.loads(info_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ConfigError(f"corrupt {info_path}: {exc}") from exc
        cached_revision = info.get("revision")
        if revision is None or cached_revision == revision:
            log.info(
                "financebench_cache_hit",
                path=str(questions_path),
                revision=cached_revision,
                n_rows=info.get("n_rows"),
            )
            return load_questions_jsonl(questions_path)
        log.info("financebench_revision_changed", cached=cached_revision, requested=revision)

    try:
        rows, info = _load_hf_rows(split, revision, cache_dir)
    except ConfigError:
        raise
    except Exception as exc:  # datasets raises many types; make the failure loud and typed
        raise ConfigError(f"could not load {DATASET_NAME} split {split!r}: {exc}") from exc
    questions = questions_from_rows(rows)
    save_questions_jsonl(questions, questions_path)
    info = {**info, "loaded_at": datetime.now(UTC).isoformat(timespec="seconds")}
    cache_dir.mkdir(parents=True, exist_ok=True)
    info_path.write_text(json.dumps(info, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    log.info(
        "financebench_loaded",
        n_questions=len(questions),
        split=split,
        revision=info.get("revision"),
        fingerprint=info.get("fingerprint"),
        path=str(questions_path),
    )
    return questions


def dataset_revision(cache_dir: Path = DEFAULT_CACHE_DIR) -> str | None:
    """The revision recorded by the last :func:`load_financebench` (``None`` if never loaded)."""
    info_path = Path(cache_dir) / DATASET_INFO_NAME
    if not info_path.is_file():
        return None
    try:
        info = json.loads(info_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    revision = info.get("revision")
    return str(revision) if revision else None


# ---------------------------------------------------------------------------------------------
# PDFs
# ---------------------------------------------------------------------------------------------


def download_pdfs(
    questions: Iterable[FBQuestion], out_dir: Path, edgar: EdgarClient | None = None
) -> DownloadReport:
    """Fetch the PDF of every document the questions reference (delegates to indexing).

    GitHub raw URL with a ``%PDF`` magic-byte check first, then the question's ``doc_link``
    through the cached EDGAR client, then ``missing``; ``MANIFEST.json`` in ``out_dir`` records
    the upstream commit SHA. Never runs in the test suite against the network.
    """
    question_list = list(questions)
    doc_names = list(dict.fromkeys(q.doc_name for q in question_list))
    doc_links = {q.doc_name: q.doc_link for q in question_list if q.doc_link}
    return download_financebench_pdfs(doc_names, Path(out_dir), edgar, doc_links=doc_links)


__all__ = [
    "DATASET_INFO_NAME",
    "DATASET_NAME",
    "DEFAULT_CACHE_DIR",
    "DEFAULT_SPLIT",
    "DownloadReport",
    "cached_questions_path",
    "dataset_revision",
    "download_pdfs",
    "evidence_from_row",
    "load_financebench",
    "load_questions_jsonl",
    "question_from_row",
    "questions_from_rows",
    "save_questions_jsonl",
]
