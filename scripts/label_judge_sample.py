#!/usr/bin/env python
"""Draw the 30-question human-labelling sample for the judge-vs-human agreement check.

Implements the protocol at the top of ``src/secqa/eval/human_labels.csv``:

* stratified by ``question_type`` over ONE run's ``predictions.jsonl`` (proportional allocation,
  largest remainder, at least one per type when the budget allows);
* drawn with ``random.Random(seed)`` over the *sorted* ``financebench_id`` of each type, so the
  sample is reproducible from the run alone;
* the sheet the annotator fills in (``--out``) has exactly the ``human_labels.csv`` header and
  contains ids only, so it can be pasted into the committed file;
* the text the annotator needs (question, gold answer, justification, our prediction) goes to a
  separate local pack (``--pack``, JSONL) that is refused inside ``results/``, ``tests/``,
  ``src/`` or ``docs/`` because FinanceBench is CC-BY-NC-4.0 and must not be committed.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

from secqa.core.contracts import EvalRecord, FBQuestion
from secqa.core.errors import ConfigError
from secqa.core.logging import get_logger
from secqa.eval.financebench import DEFAULT_CACHE_DIR, cached_questions_path, load_questions_jsonl
from secqa.eval.metrics import PREDICTIONS_NAME, read_records

log = get_logger("secqa.label_judge_sample")

DEFAULT_N = 30
DEFAULT_SEED = 0
SHEET_COLUMNS: tuple[str, ...] = (
    "run_id",
    "financebench_id",
    "label",
    "annotator",
    "labelled_at",
    "notes",
)
SHEET_NAME = "human_labels_sheet.csv"
COMMITTED_DIRS: tuple[str, ...] = ("results", "tests", "src", "docs")
REPO_ROOT = Path(__file__).resolve().parents[1]


def allocate(counts: dict[str, int], n: int) -> dict[str, int]:
    """Proportional allocation of ``n`` over strata with largest-remainder rounding.

    Every non-empty stratum gets at least one slot when ``n >= len(counts)``; no stratum is
    allocated more than it holds.
    """
    strata = {k: v for k, v in counts.items() if v > 0}
    total = sum(strata.values())
    if not strata or n <= 0:
        return {k: 0 for k in counts}
    n = min(n, total)
    quotas = {k: n * v / total for k, v in strata.items()}
    alloc = {k: int(q) for k, q in quotas.items()}
    floor = 1 if n >= len(strata) else 0
    for k in alloc:
        alloc[k] = max(alloc[k], floor)
    # Fix the total: shrink the most over-allocated strata (never below the floor), then hand
    # out remainders by fractional part.
    while sum(alloc.values()) > n:
        candidates = [k for k in alloc if alloc[k] > floor]
        k = max(candidates, key=lambda key: (alloc[key] - quotas[key], key))
        alloc[k] -= 1
    remainders = sorted(strata, key=lambda key: (-(quotas[key] - alloc[key]), key))
    for k in remainders:
        if sum(alloc.values()) >= n:
            break
        if alloc[k] < strata[k]:
            alloc[k] += 1
    for k in alloc:
        alloc[k] = min(alloc[k], strata[k])
    return {k: alloc.get(k, 0) for k in counts}


def stratified_sample(records: Sequence[EvalRecord], n: int, seed: int) -> list[EvalRecord]:
    """Deterministic stratified sample (by ``question_type``) of scored records."""
    scored = [r for r in records if not r.error]
    by_type: dict[str, list[EvalRecord]] = defaultdict(list)
    for rec in scored:
        by_type[rec.question_type].append(rec)
    for recs in by_type.values():
        recs.sort(key=lambda r: r.financebench_id)
    alloc = allocate({k: len(v) for k, v in by_type.items()}, n)
    rng = random.Random(seed)
    chosen: list[EvalRecord] = []
    for qtype in sorted(by_type):
        k = alloc[qtype]
        if k:
            chosen.extend(rng.sample(by_type[qtype], k))
    chosen.sort(key=lambda r: (r.question_type, r.financebench_id))
    log.info(
        "label_sample_drawn",
        n_requested=n,
        n_drawn=len(chosen),
        seed=seed,
        allocation={k: alloc[k] for k in sorted(alloc)},
    )
    return chosen


def write_sheet(records: Sequence[EvalRecord], out: Path, annotator: str = "") -> Path:
    """Write the ids-only labelling sheet with the ``human_labels.csv`` header."""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(SHEET_COLUMNS))
        writer.writeheader()
        for rec in records:
            writer.writerow(
                {
                    "run_id": rec.run_id,
                    "financebench_id": rec.financebench_id,
                    "label": "",
                    "annotator": annotator,
                    "labelled_at": "",
                    "notes": "",
                }
            )
    return out


def assert_private_path(path: Path) -> None:
    """Refuse to write dataset text anywhere that is committed in this repository."""
    resolved = Path(path).resolve()
    for name in COMMITTED_DIRS:
        committed = (REPO_ROOT / name).resolve()
        if resolved == committed or committed in resolved.parents:
            raise ConfigError(
                f"refusing to write dataset text under {committed} (CC-BY-NC-4.0); "
                "use a path under data/ or outside the repository"
            )


def write_pack(
    records: Sequence[EvalRecord], questions: Sequence[FBQuestion], out: Path
) -> tuple[Path, list[str]]:
    """Write the annotator pack (question, gold, justification, prediction) as local JSONL.

    Returns the path and the ids that had no matching question (skipped with a warning).
    """
    assert_private_path(out)
    by_id = {q.id: q for q in questions}
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    missing: list[str] = []
    with out.open("w", encoding="utf-8") as fh:
        for rec in records:
            q = by_id.get(rec.financebench_id)
            if q is None:
                missing.append(rec.financebench_id)
                continue
            fh.write(
                json.dumps(
                    {
                        "run_id": rec.run_id,
                        "financebench_id": rec.financebench_id,
                        "question_type": rec.question_type,
                        "question": q.question,
                        "gold_answer": q.answer,
                        "justification": q.justification,
                        "prediction": {
                            "text": rec.answer_text,
                            "value": rec.value,
                            "unit": rec.unit,
                            "abstained": rec.abstained,
                        },
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    if missing:
        log.warning("pack_questions_missing", ids=missing)
    return out, missing


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run", type=Path, required=True, help="results/<config>/<run_id>")
    parser.add_argument("--n", type=int, default=DEFAULT_N)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--annotator", default="")
    parser.add_argument("--out", type=Path, default=None, help=f"default <run>/{SHEET_NAME}")
    parser.add_argument(
        "--pack",
        type=Path,
        default=None,
        help="also write the annotator pack (dataset text) to this LOCAL path",
    )
    parser.add_argument(
        "--questions",
        type=Path,
        default=cached_questions_path(DEFAULT_CACHE_DIR),
        help="cached FinanceBench JSONL used for --pack",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Entry point; 0 on success, 2 on a configuration error."""
    args = parse_args(argv)
    if args.n < 1:
        print("[config] --n must be >= 1", file=sys.stderr)
        return 2
    pred_path = args.run / PREDICTIONS_NAME
    try:
        records = read_records(pred_path)
    except ConfigError as exc:
        print(f"[config] {exc}", file=sys.stderr)
        return 2
    sample = stratified_sample(records, args.n, args.seed)
    if not sample:
        print(f"[config] no scored records in {pred_path}", file=sys.stderr)
        return 2
    sheet = write_sheet(sample, args.out or (args.run / SHEET_NAME), args.annotator)
    print(f"sheet: {sheet} ({len(sample)} rows)")
    if args.pack is not None:
        try:
            questions = load_questions_jsonl(args.questions)
            pack, missing = write_pack(sample, questions, args.pack)
        except ConfigError as exc:
            print(f"[config] {exc}", file=sys.stderr)
            return 2
        print(f"pack: {pack} ({len(sample) - len(missing)} rows; local only, do not commit)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
