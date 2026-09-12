"""download_financebench_pdfs with every HTTP call mocked (nothing is ever really downloaded)."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
import respx

from secqa.core.errors import ConfigError
from secqa.edgar import EdgarClient
from secqa.indexing import download_financebench_pdfs, financebench_pdf_url
from secqa.indexing.financebench_corpus import (
    DOWNLOAD_USER_AGENT,
    FINANCEBENCH_COMMIT_URL,
    PDF_MANIFEST_NAME,
)

PDF_OK = b"%PDF-1.4\n1 0 obj << >> endobj\n%%EOF\n"
PDF_FALLBACK = b"%PDF-1.7\n% from the issuer site\n%%EOF\n"
FALLBACK_LINK = "https://www.sec.gov/Archives/edgar/data/1234567/000123456723000020/html.pdf"


@pytest.fixture
def routes(respx_router: respx.MockRouter) -> dict[str, respx.Route]:
    return {
        "commit": respx_router.get(FINANCEBENCH_COMMIT_URL).mock(
            return_value=httpx.Response(200, json={"sha": "abc123def"})
        ),
        "ok": respx_router.get(financebench_pdf_url("FIXTURE_2023_10K")).mock(
            return_value=httpx.Response(200, content=PDF_OK)
        ),
        "html": respx_router.get(financebench_pdf_url("HTMLONLY_2022_10K")).mock(
            return_value=httpx.Response(200, content=b"<html>Not Found</html>")
        ),
        "fallback": respx_router.get(FALLBACK_LINK).mock(
            return_value=httpx.Response(200, content=PDF_FALLBACK)
        ),
        "gone": respx_router.get(financebench_pdf_url("GONE_2020_10K")).mock(
            return_value=httpx.Response(404)
        ),
        "net": respx_router.get(financebench_pdf_url("NET_2019_10K")).mock(
            side_effect=httpx.ConnectError("boom")
        ),
    }


def test_download_writes_valid_pdfs_and_manifest(
    tmp_path: Path, routes: dict[str, respx.Route], edgar_client: EdgarClient
) -> None:
    out = tmp_path / "pdfs"
    names = [
        "FIXTURE_2023_10K",
        "HTMLONLY_2022_10K",
        "GONE_2020_10K",
        "NET_2019_10K",
        "FIXTURE_2023_10K",
    ]
    report = download_financebench_pdfs(
        names, out, edgar=edgar_client, doc_links={"HTMLONLY_2022_10K": FALLBACK_LINK}
    )
    assert report.downloaded == ["FIXTURE_2023_10K"]
    assert report.fallback == ["HTMLONLY_2022_10K"]
    assert report.cached == []
    assert report.missing == ["GONE_2020_10K", "NET_2019_10K"]
    assert "HTTP 404" in report.errors["GONE_2020_10K"]
    assert "network error" in report.errors["NET_2019_10K"]
    assert report.upstream_commit == "abc123def"
    assert report.seconds >= 0.0

    assert (out / "FIXTURE_2023_10K.pdf").read_bytes() == PDF_OK
    assert (out / "HTMLONLY_2022_10K.pdf").read_bytes() == PDF_FALLBACK
    assert not (out / "GONE_2020_10K.pdf").exists()
    assert not list(out.glob(".*.tmp"))

    manifest = json.loads((out / PDF_MANIFEST_NAME).read_text())
    assert manifest["upstream_commit"] == "abc123def"
    entry = manifest["files"]["FIXTURE_2023_10K"]
    assert entry["url"] == financebench_pdf_url("FIXTURE_2023_10K")
    assert entry["size"] == len(PDF_OK) and len(entry["sha256"]) == 64
    assert entry["fallback"] is False
    assert manifest["files"]["HTMLONLY_2022_10K"]["url"] == FALLBACK_LINK
    assert manifest["files"]["HTMLONLY_2022_10K"]["fallback"] is True
    assert "GONE_2020_10K" not in manifest["files"]

    # The GitHub fetches carry our descriptive User-Agent; the fallback went through EDGAR's.
    assert routes["ok"].calls.last.request.headers["user-agent"] == DOWNLOAD_USER_AGENT
    assert routes["fallback"].calls.last.request.headers["user-agent"] == edgar_client.user_agent


def test_download_second_run_uses_cache_without_network(
    tmp_path: Path, routes: dict[str, respx.Route], edgar_client: EdgarClient
) -> None:
    out = tmp_path / "pdfs"
    names = ["FIXTURE_2023_10K", "HTMLONLY_2022_10K"]
    download_financebench_pdfs(
        names, out, edgar=edgar_client, doc_links={"HTMLONLY_2022_10K": FALLBACK_LINK}
    )
    calls_before = routes["ok"].call_count
    report = download_financebench_pdfs(names, out)
    assert report.cached == names and report.downloaded == [] and report.missing == []
    assert routes["ok"].call_count == calls_before
    assert report.upstream_commit == "abc123def"  # preserved from the manifest


def test_download_replaces_invalid_file_and_reports_missing_fallback(
    tmp_path: Path, routes: dict[str, respx.Route]
) -> None:
    out = tmp_path / "pdfs"
    out.mkdir()
    (out / "FIXTURE_2023_10K.pdf").write_bytes(b"<html>saved error page</html>")
    report = download_financebench_pdfs(
        ["FIXTURE_2023_10K", "HTMLONLY_2022_10K"],
        out,
        doc_links={"HTMLONLY_2022_10K": FALLBACK_LINK},
    )
    assert report.downloaded == ["FIXTURE_2023_10K"]
    assert (out / "FIXTURE_2023_10K.pdf").read_bytes() == PDF_OK
    assert report.missing == ["HTMLONLY_2022_10K"]
    assert "needs an EdgarClient" in report.errors["HTMLONLY_2022_10K"]
    assert routes["fallback"].call_count == 0


def test_download_survives_commit_api_failure(
    tmp_path: Path, respx_router: respx.MockRouter
) -> None:
    respx_router.get(FINANCEBENCH_COMMIT_URL).mock(return_value=httpx.Response(403))
    respx_router.get(financebench_pdf_url("FIXTURE_2023_10K")).mock(
        return_value=httpx.Response(200, content=PDF_OK)
    )
    report = download_financebench_pdfs(["FIXTURE_2023_10K"], tmp_path / "pdfs")
    assert report.downloaded == ["FIXTURE_2023_10K"] and report.upstream_commit is None


def test_download_with_no_names_makes_no_requests(
    tmp_path: Path, respx_router: respx.MockRouter
) -> None:
    report = download_financebench_pdfs([], tmp_path / "pdfs")
    assert report.downloaded == [] and report.missing == []
    assert respx_router.calls.call_count == 0
    assert (tmp_path / "pdfs" / PDF_MANIFEST_NAME).is_file()


def test_corrupt_manifest_is_config_error(tmp_path: Path, respx_router: respx.MockRouter) -> None:
    out = tmp_path / "pdfs"
    out.mkdir()
    (out / PDF_MANIFEST_NAME).write_text("{not json", encoding="utf-8")
    with pytest.raises(ConfigError, match="corrupt"):
        download_financebench_pdfs(["FIXTURE_2023_10K"], out)
