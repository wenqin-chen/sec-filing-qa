"""scripts/verify_sources.py against respx-mocked HF, GitHub and SEC endpoints (no network)."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

import httpx
import pytest
import yaml

DOCS = ["ALPHA_2022_10K", "BETA_2023_10K", "GAMMA_2021_10K"]
COMPANIES_YAML = {
    "companies": [
        {
            "name": "Alpha Corp",
            "ticker": "ALP",
            "cik": "0000000001",
            "financebench_aliases": ["ALPHA"],
        },
        {
            "name": "Beta Inc",
            "ticker": "BET",
            "cik": "0000000002",
            "financebench_aliases": ["BETA"],
        },
        {
            "name": "Gone Ltd",
            "ticker": "GON",
            "cik": "0000000003",
            "financebench_aliases": ["GAMMA"],
        },
    ]
}


@pytest.fixture
def vs(script: Callable[[str], ModuleType]) -> ModuleType:
    return script("verify_sources")


@pytest.fixture
def companies_path(tmp_path: Path) -> Path:
    path = tmp_path / "companies.yaml"
    path.write_text(yaml.safe_dump(COMPANIES_YAML), encoding="utf-8")
    return path


def _rows_page(offset: int, length: int, total: int) -> dict[str, Any]:
    rows = []
    for i in range(offset, min(offset + length, total)):
        rows.append({"row_idx": i, "row": {"financebench_id": f"fb_{i}", "doc_name": DOCS[i % 3]}})
    return {"num_rows_total": total, "rows": rows}


def _mock_all(router: Any, vs: ModuleType, *, total: int = 150, pdf_ok: bool = True) -> None:
    router.get(vs.HF_SPLITS_URL).mock(
        return_value=httpx.Response(
            200,
            json={"splits": [{"dataset": vs.HF_DATASET, "config": "default", "split": "train"}]},
        )
    )
    router.get(url__regex=r"https://datasets-server\.huggingface\.co/rows.*").mock(
        side_effect=lambda request: httpx.Response(
            200,
            json=_rows_page(
                int(request.url.params["offset"]), int(request.url.params["length"]), total
            ),
        )
    )
    body = b"%PDF-1.7\n%synthetic" if pdf_ok else b"<!DOCTYPE html><html>404</html>"
    router.get(
        url__regex=r"https://raw\.githubusercontent\.com/patronus-ai/financebench/main/pdfs/.*\.pdf"
    ).mock(return_value=httpx.Response(200, content=body))
    router.get(vs.TICKERS_URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "0": {"cik_str": 1, "ticker": "ALP", "title": "Alpha Corp"},
                "1": {"cik_str": 2, "ticker": "BET", "title": "Beta Inc"},
            },
        )
    )
    router.get(vs.submissions_url("0000000003")).mock(
        return_value=httpx.Response(200, json={"cik": "3", "name": "Gone Ltd"})
    )


def test_all_checks_pass(
    respx_router: Any, vs: ModuleType, companies_path: Path, tmp_path: Path
) -> None:
    _mock_all(respx_router, vs)
    report = tmp_path / "report.json"
    code = vs.main(
        [
            "--companies",
            str(companies_path),
            "--pdf-sample",
            "2",
            "--user-agent",
            "Test Runner test@example.com",
            "--report",
            str(report),
        ]
    )
    assert code == 0
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["ok"] is True
    assert [c["name"] for c in payload["checks"]] == ["hf_split", "pdf_magic", "companies_ciks"]
    assert "delisted" in payload["checks"][2]["detail"]  # GON resolved through submissions


def test_wrong_row_count_fails(respx_router: Any, vs: ModuleType, companies_path: Path) -> None:
    _mock_all(respx_router, vs, total=149)
    with vs.make_client("Test Runner test@example.com") as client:
        checks = vs.run_checks(
            client, companies_path=companies_path, pdf_sample=1, sleep=lambda _s: None
        )
    assert checks[0].ok is False and "149" in checks[0].detail
    assert checks[1].ok is True, "doc names still flow to the PDF check"


def test_pdf_without_magic_bytes_fails(
    respx_router: Any, vs: ModuleType, companies_path: Path
) -> None:
    _mock_all(respx_router, vs, pdf_ok=False)
    with vs.make_client("Test Runner test@example.com") as client:
        checks = vs.run_checks(
            client, companies_path=companies_path, pdf_sample=2, sleep=lambda _s: None
        )
    assert checks[1].ok is False and "no %PDF marker" in checks[1].detail
    assert all(c.ok for c in checks if c.name != "pdf_magic")


def test_cik_mismatch_fails(respx_router: Any, vs: ModuleType, tmp_path: Path) -> None:
    _mock_all(respx_router, vs)
    bad = json.loads(json.dumps(COMPANIES_YAML))
    bad["companies"][0]["cik"] = "0000000099"
    path = tmp_path / "companies.yaml"
    path.write_text(yaml.safe_dump(bad), encoding="utf-8")
    with vs.make_client("Test Runner test@example.com") as client:
        checks = vs.run_checks(client, companies_path=path, pdf_sample=1, sleep=lambda _s: None)
    assert checks[2].ok is False
    assert "ALP: yaml 0000000099 != SEC 0000000001" in checks[2].detail


def test_main_exit_codes(respx_router: Any, vs: ModuleType, companies_path: Path) -> None:
    _mock_all(respx_router, vs, pdf_ok=False)
    assert (
        vs.main(["--companies", str(companies_path), "--user-agent", "Runner r@example.com"]) == 1
    )
    # No User-Agent anywhere (conftest scrubs SEC_USER_AGENT) -> configuration error, no network.
    assert vs.main(["--companies", str(companies_path)]) == 2
    assert vs.main(["--companies", str(companies_path), "--user-agent", "no-email-here"]) == 2
