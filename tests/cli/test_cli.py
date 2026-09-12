"""The Typer CLI end to end through ``CliRunner``: help, doctor (offline and with a mocked
vendor endpoint), ingest -> ask -> index -> export on a synthetic corpus, the mock eval row with
``--limit``, rescore, the golden report, EDGAR-backed commands over respx, and serve wiring."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx

from secqa.cli import format_value, parse_csv, parse_years
from secqa.edgar.models import TICKERS_URL, archive_url, companyfacts_url, submissions_url
from secqa.eval.financebench import cached_questions_path, load_questions_jsonl
from secqa.indexing.financebench_corpus import (
    FINANCEBENCH_COMMIT_URL,
    PDF_MANIFEST_NAME,
    financebench_pdf_url,
)
from tests.cli.conftest import (
    CONFIGS_DIR,
    EMBEDDER,
    FB_MINI,
    FIXTURES,
    GOLDEN_REPORT,
    NET_SALES_QUESTION,
    REPO_ROOT,
    TEST_UA,
    TOP_DOC,
    Invoke,
)

FIXTURE_CIK = "0001234567"


def _json(result: Any) -> Any:
    assert result.exit_code == 0, result.output
    return json.loads(result.stdout)


# ---- help / version / parsing helpers -------------------------------------------------------


def test_help_lists_every_spec_command(invoke: Invoke) -> None:
    result = invoke("--help")
    assert result.exit_code == 0
    for command in ("doctor", "data", "ingest", "index", "ask", "eval", "rescore", "report"):
        assert command in result.stdout
    for command in ("judge-swap", "human-agreement"):  # SPEC 7 judge-agreement checks
        assert command in result.stdout
    for command in ("serve", "export"):
        assert command in result.stdout
    for group, subcommands in (
        ("data", ("financebench", "companies-xbrl")),
        ("ingest", ("financebench", "ticker")),
        ("index", ("pack", "fetch", "manifest")),
    ):
        sub = invoke(group, "--help")
        assert sub.exit_code == 0
        for name in subcommands:
            assert name in sub.stdout
    version = invoke("--version")
    assert version.exit_code == 0 and version.stdout.startswith("secqa ")
    bare = invoke()
    assert "Usage" in bare.output  # no_args_is_help


def test_format_value_is_human_readable() -> None:
    assert format_value(1_577_000_000.0) == "1,577,000,000"
    assert format_value(-245.0) == "-245"
    assert format_value(0.1234) == "0.1234"
    assert format_value(12.5) == "12.5"


def test_parse_years_and_csv() -> None:
    assert parse_years("2022-2024") == [2022, 2023, 2024]
    assert parse_years("2023,2021, 2023") == [2021, 2023]
    assert parse_csv("10-K, 10-Q,") == ("10-K", "10-Q")
    with pytest.raises(ValueError, match="ends before"):
        parse_years("2024-2022")
    with pytest.raises(ValueError, match="EDGAR range"):
        parse_years("1980")
    with pytest.raises(ValueError):
        parse_years("abc")
    with pytest.raises(ValueError, match="comma-separated"):
        parse_csv(" , ")


# ---- doctor ---------------------------------------------------------------------------------


def test_doctor_passes_offline_with_no_keys(
    invoke: Invoke, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Pin the index to an unbuilt temporary path so a real data/index.duckdb on the
    # developer machine cannot turn the "index" check green.
    monkeypatch.setenv("SECQA_DUCKDB_PATH", str(tmp_path / "index.duckdb"))
    result = invoke("-q", "doctor", "--offline", "--json")
    payload = _json(result)
    assert payload["failed"] == 0
    by_name = {check["name"]: check for check in payload["checks"]}
    assert by_name["OPENAI_API_KEY"]["status"] == "skip"
    assert by_name["ANTHROPIC_API_KEY"]["status"] == "skip"
    assert by_name["SEC_USER_AGENT"]["status"] == "warn"
    assert by_name["default provider"]["status"] == "ok"
    assert by_name["price table"]["status"] == "ok"
    assert by_name["index"]["status"] == "warn"  # tmp duckdb path, not built
    assert by_name["embedder"]["status"] == "ok"
    assert by_name["duckdb fts"]["status"] in ("ok", "warn")
    assert all(check["status"] == "skip" for n, check in by_name.items() if n.startswith("model "))
    table = invoke("-q", "doctor", "--offline")
    assert table.exit_code == 0 and "0 failed" in table.stdout


def test_doctor_queries_vendor_model_ids_when_a_key_is_set(
    invoke: Invoke, monkeypatch: pytest.MonkeyPatch, respx_router: respx.MockRouter
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-a-real-key")
    seen: list[str] = []

    def reply(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers.get("authorization", ""))
        model = request.url.path.rsplit("/", 1)[-1]
        if model == "gpt-5.4-mini":
            return httpx.Response(404, json={"error": {"message": "model not found"}})
        return httpx.Response(200, json={"id": model, "object": "model"})

    respx_router.get(url__regex=r"https://api\.openai\.com/v1/models/.+").mock(side_effect=reply)
    result = invoke("-q", "doctor", "--json")
    assert result.exit_code == 1  # one model id is unknown at the vendor
    by_name = {check["name"]: check for check in json.loads(result.stdout)["checks"]}
    assert by_name["OPENAI_API_KEY"]["status"] == "ok"
    assert by_name["model openai:gpt-5.5"]["status"] == "ok"
    assert by_name["model openai:gpt-5.4-mini"]["status"] == "fail"
    assert by_name["model anthropic:claude-opus-5"]["status"] == "skip"  # no anthropic key
    assert seen and all(value == "Bearer sk-test-not-a-real-key" for value in seen)


# ---- ingest financebench -> ask -> index -> export --------------------------------------------


def test_ingest_financebench_builds_a_manifest_and_skips_unchanged(
    invoke: Invoke, built_index: Path, pdf_dir: Path, companies_path: Path
) -> None:
    payload = _json(invoke("-q", "index", "manifest", "--db", str(built_index), "--json"))
    manifest = payload["manifest"]
    assert manifest["n_documents"] == 1 and manifest["n_pages"] == 3 and manifest["n_chunks"] >= 3
    assert manifest["embedder"] == "hashing-64" and manifest["dim"] == 64
    assert len(manifest["inputs_sha256"]) == 64
    assert payload["counts"]["documents"] == 1
    again = invoke(
        "-q",
        "ingest",
        "financebench",
        "--pdf-dir",
        str(pdf_dir),
        "--questions",
        str(FB_MINI),
        "--companies",
        str(companies_path),
        "--embedder",
        EMBEDDER,
        "--db",
        str(built_index),
    )
    assert again.exit_code == 0, again.output
    assert "unchanged" in again.stdout and "1" in again.stdout
    # a different embedder must not silently mix vector spaces
    other = invoke(
        "-q",
        "ingest",
        "financebench",
        "--pdf-dir",
        str(pdf_dir),
        "--doc-name",
        TOP_DOC,
        "--companies",
        str(companies_path),
        "--embedder",
        "hashing:32",
        "--db",
        str(built_index),
    )
    assert other.exit_code == 1 and "error:" in other.stderr


def test_ingest_financebench_missing_pdf_dir_is_a_clean_error(
    invoke: Invoke, tmp_path: Path, companies_path: Path
) -> None:
    result = invoke(
        "-q",
        "ingest",
        "financebench",
        "--pdf-dir",
        str(tmp_path / "nowhere"),
        "--companies",
        str(companies_path),
        "--embedder",
        EMBEDDER,
        "--db",
        str(tmp_path / "x.duckdb"),
    )
    assert result.exit_code == 1
    assert "error: PDF directory not found" in result.stderr


def test_ask_rag_mock_returns_a_verified_citation(invoke: Invoke, built_index: Path) -> None:
    result = invoke(
        "-q",
        "ask",
        NET_SALES_QUESTION,
        "--db",
        str(built_index),
        "--embedder",
        EMBEDDER,
        "--ticker",
        "fixt",
        "--k",
        "4",
        "--json",
    )
    answer = _json(result)
    assert answer["mode"] == "rag" and answer["provider"] == "mock"
    assert answer["terminated_by"] == "single_shot" and not answer["abstained"]
    assert answer["citations"] and all(c["verified"] for c in answer["citations"])
    assert answer["citations"][0]["doc_name"] == TOP_DOC
    assert answer["citations"][0]["page_num"] == 1
    assert "1,577" in answer["text"]
    assert answer["cost_usd"] == 0.0 and answer["retrieved"]

    human = invoke(
        "-q", "ask", NET_SALES_QUESTION, "--db", str(built_index), "--embedder", EMBEDDER
    )
    assert human.exit_code == 0, human.output
    assert "verified" in human.stdout and "rag" in human.stdout


def test_ask_agent_and_closed_book_modes(invoke: Invoke, built_index: Path) -> None:
    agent = _json(
        invoke(
            "-q",
            "ask",
            NET_SALES_QUESTION,
            "--mode",
            "agent",
            "--db",
            str(built_index),
            "--embedder",
            EMBEDDER,
            "--json",
        )
    )
    assert agent["mode"] == "agent" and agent["terminated_by"] == "final_answer"
    assert agent["tool_calls"] >= 2 and agent["citations"]
    traced = invoke(
        "-q",
        "ask",
        NET_SALES_QUESTION,
        "--mode",
        "agent",
        "--trace",
        "--db",
        str(built_index),
        "--embedder",
        EMBEDDER,
    )
    assert traced.exit_code == 0 and "search_filings" in traced.stdout
    closed = _json(
        invoke(
            "-q", "ask", NET_SALES_QUESTION, "--mode", "closed_book", "--provider", "mock", "--json"
        )
    )
    assert closed["mode"] == "closed_book" and closed["retrieved"] == []


def test_ask_rejects_bad_inputs_cleanly(invoke: Invoke, tmp_path: Path) -> None:
    for args, message in (
        (("ask", "q", "--mode", "oracle"), "mode must be one of"),
        (("ask", "q", "--strategy", "magic"), "strategy must be one of"),
        (("ask", "   "), "question must not be blank"),
        (("ask", "q", "--db", str(tmp_path / "missing.duckdb")), "index not found"),
        (("ask", "q", "--mode", "closed_book", "--provider", "openai:gpt-5.5"), "OPENAI_API_KEY"),
        (("ask", "q", "--mode", "closed_book", "--provider", "nope:x"), "unknown provider"),
    ):
        result = invoke("-q", *args)
        assert result.exit_code == 1, args
        assert result.stderr.startswith("error: ") and message in result.stderr, args


def test_index_pack_fetch_manifest_round_trip(
    invoke: Invoke, built_index: Path, tmp_path: Path
) -> None:
    tarball = tmp_path / "index.tar.zst"
    packed = invoke("-q", "index", "pack", "--db", str(built_index), "--out", str(tarball))
    assert packed.exit_code == 0, packed.output
    assert tarball.is_file()
    dest = tmp_path / "fetched" / "index.duckdb"
    fetched = invoke("-q", "index", "fetch", tarball.as_uri(), "--dest", str(dest))
    assert fetched.exit_code == 0, fetched.output
    original = _json(invoke("-q", "index", "manifest", "--db", str(built_index), "--json"))
    copy = _json(invoke("-q", "index", "manifest", "--db", str(dest), "--json"))
    assert copy["manifest"] == original["manifest"] and copy["counts"] == original["counts"]
    refused = invoke("-q", "index", "fetch", tarball.as_uri(), "--dest", str(dest))
    assert refused.exit_code == 1 and "--force" in refused.stderr
    no_url = invoke("-q", "index", "fetch", "--dest", str(tmp_path / "other.duckdb"))
    assert no_url.exit_code == 1 and "SECQA_INDEX_URL" in no_url.stderr
    missing = invoke("-q", "index", "manifest", "--db", str(tmp_path / "none.duckdb"))
    assert missing.exit_code == 1 and "index not found" in missing.stderr


def test_export_writes_parquet_files(invoke: Invoke, built_index: Path, tmp_path: Path) -> None:
    out_dir = tmp_path / "parquet"
    result = invoke("-q", "export", "--db", str(built_index), "--out", str(out_dir))
    assert result.exit_code == 0, result.output
    names = {path.name for path in out_dir.glob("*.parquet")}
    assert {"documents.parquet", "pages.parquet", "chunks.parquet"} <= names


# ---- eval / rescore / report ----------------------------------------------------------------


def test_eval_mock_limit_writes_summary_and_rescore_replays(
    invoke: Invoke, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(REPO_ROOT)  # configs/rag_mock.yaml uses repo-relative fixture paths
    results = tmp_path / "results"
    fixture_db = tmp_path / "fixture.duckdb"
    result = invoke(
        "-q",
        "eval",
        "--config",
        str(CONFIGS_DIR / "rag_mock.yaml"),
        "--limit",
        "3",
        "--db",
        str(fixture_db),
        "--out",
        str(results),
        "--cassette-dir",
        str(tmp_path / "cassettes"),
    )
    assert result.exit_code == 0, result.output
    assert fixture_db.is_file()  # the mock row built its own fixture index
    run_dirs = sorted((results / "rag_mock").iterdir())
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["n"] == 3 and summary["n_completed"] == 3
    assert summary["provider"] == "mock" and summary["judge_model"] == "rule"
    assert summary["metrics"]["page_recall_10"] == 1.0
    predictions = (run_dir / "predictions.jsonl").read_text(encoding="utf-8")
    assert predictions.count("\n") == 3
    assert "What were total net sales" not in predictions  # no dataset text persisted
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    assert config["config"]["limit"] == 3 and config["n_completed"] == 3
    assert "accuracy" in result.stdout and "completed" in result.stdout

    # a second identical invocation resumes: the complete run is left alone, nothing new is made
    rerun = invoke(
        "-q",
        "eval",
        "--config",
        str(CONFIGS_DIR / "rag_mock.yaml"),
        "--limit",
        "3",
        "--db",
        str(fixture_db),
        "--out",
        str(results),
    )
    assert rerun.exit_code == 0, rerun.output
    assert len(list((results / "rag_mock").iterdir())) == 2  # complete runs are not resumed

    rescored = invoke("-q", "rescore", "--run", str(run_dir), "--db", str(fixture_db))
    assert rescored.exit_code == 0, rescored.output
    assert (run_dir / "predictions.previous.jsonl").is_file()
    after = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    assert after["rescored_at"] and after["rescore_judge"] == "rule"
    bad = invoke("-q", "rescore", "--run", str(tmp_path / "no-such-run"))
    assert bad.exit_code == 1 and "run config not found" in bad.stderr


def test_judge_swap_and_human_agreement_commands(
    invoke: Invoke, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SPEC 7: the judge-swap kappa and the judge-vs-human kappa each have a CLI entry point
    that writes the JSON `secqa report` reads, without touching the run's own verdicts."""
    monkeypatch.chdir(REPO_ROOT)
    results = tmp_path / "results"
    fixture_db = tmp_path / "fixture.duckdb"
    result = invoke(
        "-q",
        "eval",
        "--config",
        str(CONFIGS_DIR / "rag_mock.yaml"),
        "--limit",
        "3",
        "--db",
        str(fixture_db),
        "--out",
        str(results),
    )
    assert result.exit_code == 0, result.output
    run_dir = next((results / "rag_mock").iterdir())
    predictions_before = (run_dir / "predictions.jsonl").read_bytes()
    run_id = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))["run_id"]

    # judge-swap: a scripted second judge that calls every answer correct
    scenario = tmp_path / "swap_judge.yaml"
    scenario.write_text(
        "model: swap-judge\nturns:\n"
        + 3 * '  - text: \'{"label": "correct", "rationale": "agrees"}\'\n',
        encoding="utf-8",
    )
    cassettes = tmp_path / "swap_cassettes"
    swapped = invoke(
        "-q",
        "judge-swap",
        "--run",
        str(run_dir),
        "--judge",
        f"scripted:{scenario}",
        "--cassette-dir",
        str(cassettes),
        "--json",
    )
    assert swapped.exit_code == 0, swapped.output
    report_path = run_dir / "judge_swap_scripted_swap-judge.json"
    assert report_path.is_file()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["n"] == 3 and report["name_b"] == "scripted:swap-judge"
    assert report["run_id"] == run_id and report["judge_cost_usd"] == 0.0
    printed = json.loads(swapped.stdout)
    assert printed["written"] == str(report_path) and printed["kappa"] == report["kappa"]
    assert len(list(cassettes.glob("*.json"))) == 3  # the swap judge's calls are recorded
    # unlike `rescore --judge`, the run's own verdicts are untouched
    assert (run_dir / "predictions.jsonl").read_bytes() == predictions_before
    assert not (run_dir / "predictions.previous.jsonl").exists()
    refused = invoke("-q", "judge-swap", "--run", str(run_dir), "--judge", "mock")
    assert refused.exit_code == 1 and "fabricate" in refused.stderr
    missing = invoke("-q", "judge-swap", "--run", str(tmp_path / "no-such-run"))
    assert missing.exit_code == 1 and "run config not found" in missing.stderr

    # human-agreement: two labelled ids of this run, one row of another run (ignored)
    labels = tmp_path / "labels.csv"
    labels.write_text(
        "# protocol comment\n"
        "run_id,financebench_id,label,annotator,labelled_at,notes\n"
        f"{run_id},fb_mini_001,correct,wc,2026-09-11,\n"
        f"{run_id},fb_mini_002,incorrect,wc,2026-09-11,\n"
        "other_run,fb_mini_003,incorrect,wc,2026-09-11,ignored\n",
        encoding="utf-8",
    )
    human = invoke("-q", "human-agreement", "--run", str(run_dir), "--labels", str(labels))
    assert human.exit_code == 0, human.output
    for key in ("kappa", "agreement", "confusion", "disagreements", "written"):
        assert key in human.stdout
    payload = json.loads((run_dir / "human_agreement.json").read_text(encoding="utf-8"))
    assert payload["n"] == 2 and payload["name_b"] == "human" and payload["run_id"] == run_id
    assert payload["computed_at"] and "fb_mini_001" not in payload["disagreements"]
    # the shipped labels file is empty until the author labels a real run: a clean error
    empty = invoke("-q", "human-agreement", "--run", str(run_dir))
    assert empty.exit_code == 1 and "no human label" in empty.stderr


def test_eval_run_id_writes_exactly_that_directory(
    invoke: Invoke, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``--run-id`` (what eval-full.yml passes) names the directory instead of letting the
    runner mint one, so CI can publish the run it paid for without listing results/."""
    monkeypatch.chdir(REPO_ROOT)
    results = tmp_path / "results"
    # a pre-existing run whose id sorts after the new one: a naive "latest" pick would find it
    decoy = results / "rag_mock" / "ff00aaa_20260101-0000"
    decoy.mkdir(parents=True)
    (decoy / "summary.json").write_text("{}", encoding="utf-8")
    result = invoke(
        "-q",
        "eval",
        "--config",
        str(CONFIGS_DIR / "rag_mock.yaml"),
        "--limit",
        "2",
        "--db",
        str(tmp_path / "fixture.duckdb"),
        "--out",
        str(results),
        "--cassette-dir",
        str(tmp_path / "cassettes"),
        "--run-id",
        "852c7b9_20260912-0419",
    )
    assert result.exit_code == 0, result.output
    run_dir = results / "rag_mock" / "852c7b9_20260912-0419"
    assert run_dir.is_dir(), sorted(p.name for p in (results / "rag_mock").iterdir())
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["run_id"] == "852c7b9_20260912-0419" and summary["n_completed"] == 2
    assert (run_dir / "config.json").is_file() and (run_dir / "predictions.jsonl").is_file()
    assert (decoy / "summary.json").read_text(encoding="utf-8") == "{}"  # untouched

    bad = invoke(
        "-q",
        "eval",
        "--config",
        str(CONFIGS_DIR / "rag_mock.yaml"),
        "--db",
        str(tmp_path / "fixture.duckdb"),
        "--out",
        str(results),
        "--cassette-dir",
        str(tmp_path / "cassettes"),
        "--run-id",
        "a/b",
    )
    assert bad.exit_code == 1 and "plain directory name" in bad.stderr


def test_eval_rebuilds_a_stale_fixture_index(
    invoke: Invoke, tmp_path: Path, built_index: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The fixture index at --db was built with hashing-64; the config wants hashing (384)."""
    monkeypatch.chdir(REPO_ROOT)
    result = invoke(
        "-q",
        "eval",
        "--config",
        str(CONFIGS_DIR / "agent_mock.yaml"),
        "--limit",
        "1",
        "--db",
        str(built_index),
        "--out",
        str(tmp_path / "results"),
    )
    assert result.exit_code == 0, result.output
    manifest = _json(invoke("-q", "index", "manifest", "--db", str(built_index), "--json"))
    assert manifest["manifest"]["embedder"] == "hashing-384"
    assert manifest["manifest"]["n_documents"] == 2


def test_eval_without_an_index_is_a_clean_error(invoke: Invoke, tmp_path: Path) -> None:
    config = tmp_path / "row.yaml"
    config.write_text(
        "mode: rag\nprovider: mock\nembedder: hashing\njudge: rule\ncassette_mode: 'off'\n"
        f"questions_path: {FB_MINI}\n",
        encoding="utf-8",
    )
    result = invoke("-q", "eval", "--config", str(config), "--db", str(tmp_path / "no.duckdb"))
    assert result.exit_code == 1 and "index not found" in result.stderr
    unknown = invoke("-q", "eval", "--config", str(tmp_path / "missing.yaml"))
    assert unknown.exit_code == 1 and "error:" in unknown.stderr


def test_report_renders_the_golden_document(invoke: Invoke, tmp_path: Path) -> None:
    out_path = tmp_path / "RESULTS.md"
    result = invoke(
        "-q",
        "report",
        str(tmp_path / "results"),
        "--out",
        str(out_path),
        "--configs",
        str(CONFIGS_DIR),
    )
    assert result.exit_code == 0, result.output
    rendered = out_path.read_text(encoding="utf-8")
    expected = GOLDEN_REPORT.read_text(encoding="utf-8")

    def without_stamp(text: str) -> list[str]:
        return [line for line in text.splitlines() if not line.startswith("Generated by")]

    assert without_stamp(rendered) == without_stamp(expected)
    assert "pending (not run)" in rendered and "| `rag_mock`" not in rendered
    stdout = invoke(
        "-q", "report", str(tmp_path / "results"), "--out", "-", "--configs", str(CONFIGS_DIR)
    )
    assert stdout.exit_code == 0 and stdout.stdout == rendered
    missing = invoke("-q", "report", "--configs", str(tmp_path / "no-configs"))
    assert missing.exit_code == 1 and "not found" in missing.stderr


# ---- data (Hugging Face cache + PDFs) and EDGAR-backed commands ------------------------------


def _seed_financebench_cache(cache_dir: Path) -> None:
    questions = load_questions_jsonl(FB_MINI)
    from secqa.eval.financebench import save_questions_jsonl

    save_questions_jsonl(questions, cached_questions_path(cache_dir))
    (cache_dir / "DATASET.json").write_text(
        json.dumps({"revision": "main", "n_rows": len(questions), "split": "train"}),
        encoding="utf-8",
    )


def test_data_financebench_reads_the_cache_and_downloads_pdfs(
    invoke: Invoke, tmp_path: Path, pdf_dir: Path, respx_router: respx.MockRouter
) -> None:
    cache_dir = tmp_path / "cache"
    _seed_financebench_cache(cache_dir)
    listed = invoke("-q", "data", "financebench", "--cache-dir", str(cache_dir))
    assert listed.exit_code == 0, listed.output
    assert "6" in listed.stdout and "metrics-generated" in listed.stdout

    pdf_bytes = (pdf_dir / f"{TOP_DOC}.pdf").read_bytes()
    respx_router.get(FINANCEBENCH_COMMIT_URL).mock(
        return_value=httpx.Response(200, json={"sha": "c" * 40})
    )
    respx_router.get(financebench_pdf_url(TOP_DOC)).mock(
        return_value=httpx.Response(200, content=pdf_bytes)
    )
    respx_router.get(financebench_pdf_url("OTHER_2022_10K")).mock(
        return_value=httpx.Response(404)  # upstream has no such file; no doc_link either
    )
    fetched = invoke("-q", "data", "financebench", "--cache-dir", str(cache_dir), "--pdfs")
    assert fetched.exit_code == 0, fetched.output
    assert (cache_dir / "pdfs" / f"{TOP_DOC}.pdf").read_bytes() == pdf_bytes
    manifest = json.loads((cache_dir / "pdfs" / PDF_MANIFEST_NAME).read_text(encoding="utf-8"))
    assert manifest["upstream_commit"] == "c" * 40
    assert "SEC_USER_AGENT is not set" in fetched.stderr  # doc_link fallback disabled
    assert "missing: OTHER_2022_10K: HTTP 404" in fetched.stderr
    assert "downloaded" in fetched.stdout


_FILINGS = [
    # accession, filingDate, reportDate, form, primaryDocument
    ("0001234567-24-000010", "2024-02-15", "2023-12-31", "10-K", "fixt-20231231.htm"),
    ("0001234567-23-000020", "2023-02-15", "2022-12-31", "10-K", "fixt-20221231.htm"),
    ("0001234567-23-000070", "2023-08-01", "2023-06-30", "10-Q", "fixt-20230630.htm"),
]


def _edgar_routes(router: respx.MockRouter) -> None:
    cols = list(zip(*_FILINGS, strict=True))
    submissions = {
        "cik": "1234567",
        "name": "Fixture Corp",
        "tickers": ["FIXT"],
        "filings": {
            "recent": {
                "accessionNumber": list(cols[0]),
                "filingDate": list(cols[1]),
                "reportDate": list(cols[2]),
                "form": list(cols[3]),
                "primaryDocument": list(cols[4]),
            },
            "files": [],
        },
    }
    router.get(TICKERS_URL).mock(
        return_value=httpx.Response(
            200, json={"0": {"cik_str": 1234567, "ticker": "FIXT", "title": "Fixture Corp"}}
        )
    )
    router.get(submissions_url(FIXTURE_CIK)).mock(
        return_value=httpx.Response(200, json=submissions)
    )
    facts = json.loads((FIXTURES / "indexing_companyfacts_small.json").read_text(encoding="utf-8"))
    router.get(companyfacts_url(FIXTURE_CIK)).mock(return_value=httpx.Response(200, json=facts))
    html = (FIXTURES / "indexing_edgar_10k.html").read_bytes()
    for accession, _filed, report, _form, doc in _FILINGS:
        body = html.replace(b"December 31.", f"December 31 ({report}).".encode())
        router.get(archive_url(FIXTURE_CIK, accession, doc)).mock(
            return_value=httpx.Response(200, content=body, headers={"content-type": "text/html"})
        )


def test_ingest_ticker_over_mocked_edgar(
    invoke: Invoke,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    respx_router: respx.MockRouter,
) -> None:
    db = tmp_path / "edgar.duckdb"
    without_ua = invoke("-q", "ingest", "ticker", "FIXT", "--db", str(db), "--embedder", EMBEDDER)
    assert without_ua.exit_code == 1 and "SEC_USER_AGENT" in without_ua.stderr

    monkeypatch.setenv("SEC_USER_AGENT", TEST_UA)
    _edgar_routes(respx_router)
    result = invoke(
        "-q",
        "ingest",
        "ticker",
        "fixt",
        "--forms",
        "10-K,10-Q",
        "--years",
        "2022-2023",
        "--embedder",
        EMBEDDER,
        "--db",
        str(db),
        "--edgar-cache-dir",
        str(tmp_path / "edgar-cache"),
    )
    assert result.exit_code == 0, result.output
    assert "FIXT_2023_10-K" in result.stdout and "FIXT_2023Q2_10-Q" in result.stdout
    manifest = _json(invoke("-q", "index", "manifest", "--db", str(db), "--json"))
    assert manifest["manifest"]["n_documents"] == 3
    assert manifest["manifest"]["n_facts"] > 0  # --xbrl default loaded companyfacts
    assert manifest["counts"]["facts"] == manifest["manifest"]["n_facts"]

    answer = _json(
        invoke(
            "-q",
            "ask",
            "What was the revenue?",
            "--ticker",
            "FIXT",
            "--db",
            str(db),
            "--embedder",
            EMBEDDER,
            "--json",
        )
    )
    assert answer["retrieved"] and all(
        v["doc_name"].startswith("FIXT_") for v in answer["retrieved"]
    )


def test_data_companies_xbrl_over_mocked_edgar(
    invoke: Invoke,
    built_index: Path,
    companies_path: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    respx_router: respx.MockRouter,
) -> None:
    monkeypatch.setenv("SEC_USER_AGENT", TEST_UA)
    _edgar_routes(respx_router)
    respx_router.get(companyfacts_url("0007654321")).mock(return_value=httpx.Response(404))
    result = invoke(
        "-q",
        "data",
        "companies-xbrl",
        "--companies",
        str(companies_path),
        "--db",
        str(built_index),
        "--edgar-cache-dir",
        str(tmp_path / "edgar-cache"),
    )
    assert result.exit_code == 0, result.output
    manifest = _json(invoke("-q", "index", "manifest", "--db", str(built_index), "--json"))
    assert manifest["manifest"]["n_facts"] > 0 and manifest["manifest"]["n_documents"] == 1
    # the curated view exists and is queryable through the read-only guard
    from secqa.store import DuckDBStore
    from secqa.xbrl import run_readonly_sql

    with DuckDBStore(built_index, read_only=True) as store:
        rows = run_readonly_sql(store, "SELECT ticker, fiscal_year FROM financials")
    assert rows.row_count > 0 and rows.rows[0][0] == "FIXT"


def test_data_companies_xbrl_needs_an_initialised_index_or_an_embedder(
    invoke: Invoke, companies_path: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SEC_USER_AGENT", TEST_UA)
    bad = tmp_path / "bad.yaml"
    bad.write_text("companies: []\n", encoding="utf-8")
    result = invoke(
        "-q", "data", "companies-xbrl", "--companies", str(bad), "--db", str(tmp_path / "n.duckdb")
    )
    assert result.exit_code == 1 and "companies" in result.stderr


# ---- serve ----------------------------------------------------------------------------------


def test_serve_wires_uvicorn_with_the_app_factory(
    invoke: Invoke, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import uvicorn

    captured: dict[str, Any] = {}

    def fake_run(target: str, **kwargs: Any) -> None:
        captured["target"] = target
        captured.update(kwargs)

    monkeypatch.setattr(uvicorn, "run", fake_run)
    db = tmp_path / "served.duckdb"
    result = invoke("-q", "serve", "--host", "127.0.0.1", "--port", "9999", "--db", str(db))
    assert result.exit_code == 0, result.output
    assert captured["target"] == "secqa.api.app:create_app" and captured["factory"] is True
    assert captured["host"] == "127.0.0.1" and captured["port"] == 9999
    assert captured["reload"] is False and captured["workers"] == 1
    assert Path(captured and __import__("os").environ["SECQA_DUCKDB_PATH"]) == db
    assert "http://127.0.0.1:9999" in result.stdout

    monkeypatch.setenv("SECQA_PROVIDER", "openai:gpt-5.5")  # needs a key the env does not have
    refused = invoke("-q", "serve")
    assert refused.exit_code == 1 and "OPENAI_API_KEY" in refused.stderr
    assert captured["port"] == 9999  # uvicorn was not started a second time
