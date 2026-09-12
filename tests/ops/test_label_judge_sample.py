"""scripts/label_judge_sample.py: stratified, seeded sample; ids-only sheet; pack path guard."""

from __future__ import annotations

import csv
import json
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import pytest

from secqa.core.contracts import EvalRecord, Evidence, FBQuestion
from secqa.core.errors import ConfigError
from secqa.eval.judge import read_human_labels
from tests.ops.conftest import REPO


@pytest.fixture
def ljs(script: Callable[[str], ModuleType]) -> ModuleType:
    return script("label_judge_sample")


@pytest.fixture
def records(eval_record: Callable[..., EvalRecord]) -> list[EvalRecord]:
    out: list[EvalRecord] = []
    for i in range(12):
        out.append(eval_record(financebench_id=f"fb_m_{i:03d}", question_type="metrics-generated"))
    for i in range(6):
        out.append(eval_record(financebench_id=f"fb_d_{i:03d}", question_type="domain-relevant"))
    for i in range(2):
        out.append(eval_record(financebench_id=f"fb_n_{i:03d}", question_type="novel-generated"))
    out.append(eval_record(financebench_id="fb_err", question_type="novel-generated", error="boom"))
    return out


@pytest.fixture
def run_dir(records: list[EvalRecord], tmp_path: Path) -> Path:
    run = tmp_path / "results" / "rag_mock" / "abc1234_20260911-1200"
    run.mkdir(parents=True)
    with (run / "predictions.jsonl").open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec.model_dump(mode="json")) + "\n")
    return run


def test_allocate_is_proportional_with_floor(ljs: ModuleType) -> None:
    assert ljs.allocate({"a": 12, "b": 6, "c": 2}, 10) == {"a": 6, "b": 3, "c": 1}
    assert ljs.allocate({"a": 12, "b": 6, "c": 2}, 30) == {"a": 12, "b": 6, "c": 2}  # capped
    assert ljs.allocate({"a": 5, "b": 0}, 3) == {"a": 3, "b": 0}
    assert ljs.allocate({"a": 5}, 0) == {"a": 0}
    alloc = ljs.allocate({"a": 100, "b": 1, "c": 1}, 5)
    assert sum(alloc.values()) == 5 and alloc["b"] == 1 and alloc["c"] == 1


def test_stratified_sample_deterministic_and_error_free(
    ljs: ModuleType, records: list[EvalRecord]
) -> None:
    a = ljs.stratified_sample(records, 10, seed=0)
    b = ljs.stratified_sample(records, 10, seed=0)
    assert [r.financebench_id for r in a] == [r.financebench_id for r in b]
    assert len(a) == 10
    by_type = {t: sum(1 for r in a if r.question_type == t) for t in {r.question_type for r in a}}
    assert by_type == {"metrics-generated": 6, "domain-relevant": 3, "novel-generated": 1}
    assert all(r.error is None for r in a)
    assert [r.financebench_id for r in ljs.stratified_sample(records, 10, seed=1)] != [
        r.financebench_id for r in a
    ]


def test_sheet_matches_human_labels_header_and_is_readable(
    ljs: ModuleType, records: list[EvalRecord], tmp_path: Path
) -> None:
    sample = ljs.stratified_sample(records, 5, seed=0)
    out = ljs.write_sheet(sample, tmp_path / "sheet.csv", annotator="wc")
    with out.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert list(rows[0]) == list(ljs.SHEET_COLUMNS)
    assert len(rows) == 5 and all(r["label"] == "" and r["annotator"] == "wc" for r in rows)
    # The eval module's reader accepts the sheet once labels are filled in.
    filled = tmp_path / "filled.csv"
    text = out.read_text(encoding="utf-8").replace(",,wc,", ",correct,wc,")
    filled.write_text(text, encoding="utf-8")
    assert len(read_human_labels(filled)) == 5


def test_pack_refuses_committed_directories(ljs: ModuleType, records: list[EvalRecord]) -> None:
    for sub in ("results/x.jsonl", "tests/fixtures/x.jsonl", "src/secqa/x.jsonl", "docs/x.jsonl"):
        with pytest.raises(ConfigError, match="refusing"):
            ljs.write_pack(records[:1], [], REPO / sub)


def test_pack_joins_questions_and_reports_missing(
    ljs: ModuleType, records: list[EvalRecord], tmp_path: Path
) -> None:
    q = FBQuestion(
        id="fb_m_000",
        company="Synthetic",
        doc_name="FIXTURE_2023_10K",
        question_type="metrics-generated",
        question="synthetic question?",
        answer="42",
        justification="because",
        evidence=[Evidence(doc_name="FIXTURE_2023_10K", page_num=1, text="x")],
    )
    sample = [r for r in records if r.financebench_id in ("fb_m_000", "fb_m_001")]
    pack, missing = ljs.write_pack(sample, [q], tmp_path / "data" / "pack.jsonl")
    lines = [json.loads(ln) for ln in pack.read_text(encoding="utf-8").splitlines()]
    assert missing == ["fb_m_001"]
    assert len(lines) == 1 and lines[0]["question"] == "synthetic question?"
    assert lines[0]["prediction"]["value"] == 1577.0


def test_main_writes_sheet_into_run_dir(
    ljs: ModuleType, run_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert ljs.main(["--run", str(run_dir), "--n", "6"]) == 0
    sheet = run_dir / ljs.SHEET_NAME
    assert sheet.is_file()
    assert "6 rows" in capsys.readouterr().out
    assert "synthetic" not in sheet.read_text(encoding="utf-8").lower()  # ids only
    assert ljs.main(["--run", str(run_dir / "missing"), "--n", "6"]) == 2
    assert ljs.main(["--run", str(run_dir), "--n", "0"]) == 2
