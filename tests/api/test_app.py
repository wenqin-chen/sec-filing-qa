"""End-to-end tests of the FastAPI service over a seeded temporary index (offline)."""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from secqa.api import create_app
from secqa.api.errors import PROBLEM_MEDIA_TYPE
from secqa.core.errors import ConfigError
from secqa.core.settings import Settings
from tests.api.conftest import (
    NET_SALES_QUESTION,
    NET_SALES_SENTENCE,
    OTHER_DOC,
    PAID_SPEC,
    TICKER,
    TOP_DOC,
    FailingProvider,
    PricedScripted,
    SlowAbstainingProvider,
    abstain_json,
    install_paid_provider,
    net_sales_chunk_id,
    rag_answer_turn,
    search_turn,
    state_of,
)

SettingsFactory = Callable[..., Settings]
ClientFactory = Callable[[FastAPI], TestClient]
BIG_USAGE = {"input_tokens": 100_000, "output_tokens": 1_000}  # $0.416 per call at gpt-test
SMALL_USAGE = {"input_tokens": 1_000, "output_tokens": 100}  # $0.0056 per call at gpt-test


def assert_problem(response: Any, status: int) -> dict[str, Any]:
    """Assert an ``application/problem+json`` body with the required members; return it."""
    assert response.status_code == status, response.text
    assert response.headers["content-type"].startswith(PROBLEM_MEDIA_TYPE)
    body = response.json()
    assert body["status"] == status
    assert body["type"].startswith("urn:secqa:problem:")
    assert body["title"] and body["detail"]
    assert body["request_id"] == response.headers["X-Request-ID"]
    return body


# ---- probes and version ----------------------------------------------------------------------


def test_healthz_needs_nothing(app: FastAPI) -> None:
    response = TestClient(app).get("/healthz")  # no lifespan: index never opened
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    assert len(response.headers["X-Request-ID"]) == 32
    assert "app;dur=" in response.headers["Server-Timing"]


def test_readyz_503_before_startup_then_200_with_counts(app: FastAPI) -> None:
    cold = TestClient(app)
    body = assert_problem(cold.get("/readyz"), 503)
    assert body["title"] == "Service not ready"
    with cold as warm:
        response = warm.get("/readyz")
        assert response.status_code == 200
        assert response.json() == {
            "status": "ready",
            "chunks": 20,
            "documents": 2,
            "facts": response.json()["facts"],
            "embedder": "hashing-64",
            "provider": "mock:mock-extractive",
        }
        assert response.json()["facts"] > 0
    assert_problem(cold.get("/readyz"), 503)  # shut down: handles released


def test_readyz_explains_a_missing_index(
    make_settings: SettingsFactory, tmp_path: Any, make_client: ClientFactory
) -> None:
    missing = tmp_path / "nowhere" / "index.duckdb"
    client = make_client(create_app(make_settings(duckdb_path=missing)))
    body = assert_problem(client.get("/readyz"), 503)
    assert "index not found" in body["detail"] and str(missing) in body["detail"]
    assert client.get("/healthz").status_code == 200
    assert_problem(client.post("/v1/ask", json={"question": NET_SALES_QUESTION}), 503)


def test_readyz_explains_an_embedder_mismatch(
    make_settings: SettingsFactory, make_client: ClientFactory
) -> None:
    client = make_client(create_app(make_settings(embedder="hashing:32")))
    body = assert_problem(client.get("/readyz"), 503)
    assert "IndexMismatch" in body["detail"]


def test_version_reports_prompts_and_index_manifest(client: TestClient) -> None:
    body = client.get("/version").json()
    assert body["version"]
    assert body["git_sha"]
    assert set(body["prompt_hashes"]) >= {
        "rag_system.md",
        "closed_book_system.md",
        "oracle_system.md",
        "agent_system.md",
    }
    assert all(len(sha) == 64 for sha in body["prompt_hashes"].values())
    assert body["index_manifest"]["n_chunks"] == 20
    assert body["index_manifest"]["embedder"] == "hashing-64"
    assert body["index_manifest"]["git_sha"] == "f" * 40


def test_create_app_refuses_a_provider_without_its_key(make_settings: SettingsFactory) -> None:
    with pytest.raises(ConfigError, match="OPENAI_API_KEY"):
        create_app(make_settings(provider="openai:gpt-5.5"))


# ---- /v1/ask ---------------------------------------------------------------------------------


def test_ask_rag_returns_verified_citations(client: TestClient) -> None:
    response = client.post(
        "/v1/ask", json={"question": NET_SALES_QUESTION, "ticker": TICKER, "k": 4}
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["mode"] == "rag"
    assert body["provider"] == "mock" and body["model"] == "mock-extractive"
    assert body["request_id"] == response.headers["X-Request-ID"]
    assert body["text"] == NET_SALES_SENTENCE
    assert body["abstained"] is False and body["grounded"] is True
    assert body["citations"], "the extractive mock cites the passage it quotes"
    assert all(c["verified"] and c["valid"] for c in body["citations"])
    top = body["citations"][0]
    assert top["doc_name"] == TOP_DOC and top["page_num"] == 1
    assert top["chunk_id"] == net_sales_chunk_id()
    assert top["snippet"].startswith("Total net sales")
    assert {hit["doc_name"] for hit in body["retrieved"]} == {TOP_DOC}
    assert body["terminated_by"] == "single_shot"
    assert body["usage"]["input_tokens"] > 0 and body["cost_usd"] == 0.0
    assert [step["kind"] for step in body["trace"]] == ["retrieval", "llm", "verify"]
    timing = response.headers["Server-Timing"]
    assert "retrieval;dur=" in timing and "llm;dur=" in timing and "app;dur=" in timing


def test_ask_echoes_a_caller_request_id(client: TestClient) -> None:
    response = client.post(
        "/v1/ask",
        json={"question": NET_SALES_QUESTION},
        headers={"X-Request-ID": "trace-abc.123"},
    )
    assert response.status_code == 200
    assert response.headers["X-Request-ID"] == "trace-abc.123"
    assert response.json()["request_id"] == "trace-abc.123"


def test_ask_include_trace_false_drops_the_trace_only(client: TestClient) -> None:
    body = client.post(
        "/v1/ask", json={"question": NET_SALES_QUESTION, "include_trace": False}
    ).json()
    assert body["trace"] == []
    assert body["citations"] and body["retrieved"]


def test_ask_closed_book_has_no_citations(client: TestClient) -> None:
    body = client.post(
        "/v1/ask", json={"question": NET_SALES_QUESTION, "mode": "closed_book"}
    ).json()
    assert body["mode"] == "closed_book"
    assert body["citations"] == [] and body["retrieved"] == []


def test_ask_agent_mode_open_without_server_key(client: TestClient) -> None:
    response = client.post("/v1/ask", json={"question": NET_SALES_QUESTION, "mode": "agent"})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["mode"] == "agent"
    assert body["terminated_by"] == "final_answer"
    assert body["tool_calls"] >= 1
    assert body["citations"] and all(c["verified"] for c in body["citations"])


def test_ask_doc_filter_matching_nothing_abstains_without_llm(client: TestClient) -> None:
    body = client.post(
        "/v1/ask", json={"question": NET_SALES_QUESTION, "doc_names": ["NOPE_2020_10K"]}
    ).json()
    assert body["abstained"] is True
    assert body["terminated_by"] == "empty_retrieval"
    assert body["usage"] == {
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_read_tokens": 0,
        "cache_write_tokens": 0,
    }


class TestApiKeyGate:
    """``SECQA_API_KEY`` set: agent mode and non-default providers need ``X-API-Key``."""

    @pytest.fixture
    def gated(self, make_settings: SettingsFactory, make_client: ClientFactory) -> TestClient:
        return make_client(create_app(make_settings(api_key="demo-secret")))

    def test_agent_mode_403_without_key(self, gated: TestClient) -> None:
        body = assert_problem(
            gated.post("/v1/ask", json={"question": NET_SALES_QUESTION, "mode": "agent"}), 403
        )
        assert "agent mode" in body["detail"] and "X-API-Key" in body["detail"]

    def test_agent_mode_403_with_wrong_key(self, gated: TestClient) -> None:
        response = gated.post(
            "/v1/ask",
            json={"question": NET_SALES_QUESTION, "mode": "agent"},
            headers={"X-API-Key": "wrong"},
        )
        assert_problem(response, 403)

    def test_agent_mode_200_with_key(self, gated: TestClient) -> None:
        response = gated.post(
            "/v1/ask",
            json={"question": NET_SALES_QUESTION, "mode": "agent"},
            headers={"X-API-Key": "demo-secret"},
        )
        assert response.status_code == 200, response.text
        assert response.json()["mode"] == "agent"

    def test_non_default_provider_403_without_key(self, gated: TestClient) -> None:
        response = gated.post(
            "/v1/ask", json={"question": NET_SALES_QUESTION, "provider": "mock:abstain"}
        )
        body = assert_problem(response, 403)
        assert "mock:abstain" in body["detail"]

    def test_default_provider_and_rag_stay_open(self, gated: TestClient) -> None:
        response = gated.post("/v1/ask", json={"question": NET_SALES_QUESTION, "provider": "mock"})
        assert response.status_code == 200


class TestAskValidation:
    def test_unknown_provider_422(self, client: TestClient) -> None:
        body = assert_problem(
            client.post("/v1/ask", json={"question": NET_SALES_QUESTION, "provider": "foo:bar"}),
            422,
        )
        assert "unknown provider 'foo:bar'" in body["detail"]

    def test_scripted_provider_not_allowed_over_http(self, client: TestClient) -> None:
        response = client.post(
            "/v1/ask", json={"question": NET_SALES_QUESTION, "provider": "scripted:/etc/passwd"}
        )
        assert_problem(response, 422)

    def test_unkeyed_vendor_provider_503(self, client: TestClient) -> None:
        body = assert_problem(
            client.post(
                "/v1/ask", json={"question": NET_SALES_QUESTION, "provider": "openai:gpt-5.5"}
            ),
            503,
        )
        assert body["title"] == "Provider not configured"
        assert "OPENAI_API_KEY" in body["detail"]

    def test_k_above_schema_bound_422(self, client: TestClient) -> None:
        body = assert_problem(
            client.post("/v1/ask", json={"question": NET_SALES_QUESTION, "k": 21}), 422
        )
        assert body["title"] == "Validation error"
        assert body["errors"][0]["loc"] == ["body", "k"]

    def test_k_above_server_max_k_422(
        self, make_settings: SettingsFactory, make_client: ClientFactory
    ) -> None:
        client = make_client(create_app(make_settings(max_k=5)))
        body = assert_problem(
            client.post("/v1/ask", json={"question": NET_SALES_QUESTION, "k": 6}), 422
        )
        assert "maximum of 5" in body["detail"]
        assert (
            client.post("/v1/ask", json={"question": NET_SALES_QUESTION, "k": 5}).status_code == 200
        )

    @pytest.mark.parametrize(
        "body",
        [
            {},
            {"question": "   "},
            {"question": "x" * 2001},
            {"question": "q", "mode": "oracle"},
            {"question": "q", "ticker": "NOT A TICKER"},
            {"question": "q", "fiscal_year": 1800},
            {"question": "q", "max_cost_usd": -1},
            {"question": "q", "doc_names": []},
            {"question": "q", "unknown_field": 1},
        ],
        ids=[
            "missing-question",
            "blank-question",
            "too-long",
            "oracle-mode",
            "bad-ticker",
            "bad-year",
            "negative-cap",
            "empty-doc-names",
            "unknown-field",
        ],
    )
    def test_invalid_bodies_422(self, client: TestClient, body: dict[str, Any]) -> None:
        problem = assert_problem(client.post("/v1/ask", json=body), 422)
        assert problem["errors"]

    def test_unknown_route_is_problem_json(self, client: TestClient) -> None:
        assert_problem(client.get("/v1/nope"), 404)
        assert_problem(client.delete("/v1/ask"), 405)


class TestBudget:
    def test_zero_cap_refused_before_paying(self, app: FastAPI, make_client: ClientFactory) -> None:
        client = make_client(app)
        provider = PricedScripted([rag_answer_turn(SMALL_USAGE)])
        install_paid_provider(app, provider)
        body = assert_problem(
            client.post(
                "/v1/ask",
                json={"question": NET_SALES_QUESTION, "provider": PAID_SPEC, "max_cost_usd": 0},
            ),
            402,
        )
        assert "max_cost_usd=0" in body["detail"]
        assert provider.turns_consumed == 0, "no paid call is made under a zero cap"
        assert state_of(app).spent_today_usd == 0.0

    def test_zero_cap_is_fine_for_the_free_mock(self, client: TestClient) -> None:
        response = client.post("/v1/ask", json={"question": NET_SALES_QUESTION, "max_cost_usd": 0})
        assert response.status_code == 200

    def test_rag_over_cap_402_with_partial_trace(
        self, app: FastAPI, make_client: ClientFactory
    ) -> None:
        client = make_client(app)
        install_paid_provider(app, PricedScripted([rag_answer_turn(BIG_USAGE)]))
        response = client.post(
            "/v1/ask",
            json={"question": NET_SALES_QUESTION, "provider": PAID_SPEC, "max_cost_usd": 0.01},
        )
        body = assert_problem(response, 402)
        assert "exceeded the cap of $0.0100" in body["detail"]
        partial = body["answer"]
        assert partial["cost_usd"] == pytest.approx(0.416)
        assert [step["kind"] for step in partial["trace"]] == ["retrieval", "llm", "verify"]
        assert partial["citations"][0]["verified"] is True
        assert state_of(app).spent_today_usd == pytest.approx(0.416), "the spend is still recorded"

    def test_agent_cost_cap_abort_402_with_partial_trace(
        self, app: FastAPI, make_client: ClientFactory
    ) -> None:
        client = make_client(app)
        provider = PricedScripted(
            [
                search_turn("total net sales fiscal 2023", BIG_USAGE),
                {"match": "Stop using tools", "text": abstain_json(), "usage": BIG_USAGE},
            ]
        )
        install_paid_provider(app, provider)
        response = client.post(
            "/v1/ask",
            json={
                "question": NET_SALES_QUESTION,
                "provider": PAID_SPEC,
                "mode": "agent",
                "max_cost_usd": 0.5,
            },
        )
        body = assert_problem(response, 402)
        assert "projected cost" in body["detail"]
        partial = body["answer"]
        assert partial["terminated_by"] == "budget" and partial["mode"] == "agent"
        assert any(step["kind"] == "tool" for step in partial["trace"])
        assert provider.turns_consumed == 2

    def test_request_cap_is_clamped_to_the_server_cap(
        self, make_settings: SettingsFactory, make_client: ClientFactory
    ) -> None:
        app = create_app(make_settings(max_cost_usd=0.1))
        client = make_client(app)
        install_paid_provider(app, PricedScripted([rag_answer_turn(BIG_USAGE)]))
        response = client.post(
            "/v1/ask",
            json={"question": NET_SALES_QUESTION, "provider": PAID_SPEC, "max_cost_usd": 100.0},
        )
        body = assert_problem(response, 402)
        assert "cap of $0.1000" in body["detail"]

    def test_daily_budget_exhausted_402_before_the_call(
        self, make_settings: SettingsFactory, make_client: ClientFactory
    ) -> None:
        app = create_app(make_settings(daily_budget_usd=0.4, max_cost_usd=1.0))
        client = make_client(app)
        provider = PricedScripted([rag_answer_turn(BIG_USAGE), rag_answer_turn(BIG_USAGE)])
        install_paid_provider(app, provider)
        first = client.post("/v1/ask", json={"question": NET_SALES_QUESTION, "provider": PAID_SPEC})
        assert first.status_code == 200, first.text
        assert state_of(app).spent_today_usd == pytest.approx(0.416)
        second = client.post(
            "/v1/ask", json={"question": NET_SALES_QUESTION, "provider": PAID_SPEC}
        )
        body = assert_problem(second, 402)
        assert "daily budget of $0.40" in body["detail"]
        assert provider.turns_consumed == 1
        assert client.post("/v1/ask", json={"question": NET_SALES_QUESTION}).status_code == 200

    def test_cheap_paid_request_succeeds_and_is_billed(
        self, app: FastAPI, make_client: ClientFactory
    ) -> None:
        client = make_client(app)
        install_paid_provider(app, PricedScripted([rag_answer_turn(SMALL_USAGE)]))
        response = client.post(
            "/v1/ask", json={"question": NET_SALES_QUESTION, "provider": PAID_SPEC}
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["provider"] == "openai" and body["model"] == "gpt-test"
        assert body["cost_usd"] == pytest.approx(0.0056)
        assert body["citations"][0]["verified"] is True


def test_provider_error_maps_to_502(app: FastAPI, make_client: ClientFactory) -> None:
    client = make_client(app)
    install_paid_provider(app, FailingProvider(retryable=True))
    response = client.post("/v1/ask", json={"question": NET_SALES_QUESTION, "provider": PAID_SPEC})
    body = assert_problem(response, 502)
    assert body["title"] == "Upstream provider error"
    assert body["retryable"] is True
    assert response.headers["Retry-After"] == "5"


def test_agent_wall_clock_abort_maps_to_504(
    make_settings: SettingsFactory, make_client: ClientFactory
) -> None:
    app = create_app(make_settings(request_timeout_s=1))
    client = make_client(app)
    provider = SlowAbstainingProvider(delay_s=1.05)
    install_paid_provider(app, provider)
    response = client.post(
        "/v1/ask", json={"question": NET_SALES_QUESTION, "provider": PAID_SPEC, "mode": "agent"}
    )
    body = assert_problem(response, 504)
    assert "wall clock" in body["detail"]
    assert body["answer"]["terminated_by"] == "budget"
    assert body["answer"]["abstained"] is True


def test_rate_limit_429_problem_json(
    make_settings: SettingsFactory, make_client: ClientFactory
) -> None:
    client = make_client(create_app(make_settings(rate_limit_per_min=2)))
    payload = {"query": "net sales", "k": 2}
    assert client.post("/v1/search", json=payload).status_code == 200
    assert client.post("/v1/search", json=payload).status_code == 200
    third = client.post("/v1/search", json=payload)
    body = assert_problem(third, 429)
    assert "2 per 1 minute" in body["detail"]
    assert "Retry-After" in third.headers
    assert client.get("/healthz").status_code == 200, "probes are never rate limited"


# ---- /v1/search, /v1/filings, /v1/xbrl/query -------------------------------------------------


class TestSearch:
    def test_hybrid_hits_are_store_sourced_views(self, client: TestClient) -> None:
        response = client.post("/v1/search", json={"query": "total net sales", "k": 3})
        assert response.status_code == 200, response.text
        hits = response.json()["hits"]
        assert 1 <= len(hits) <= 3
        assert set(hits[0]) == {"chunk_id", "doc_name", "page_num", "section", "score", "snippet"}
        assert hits[0]["chunk_id"] == net_sales_chunk_id()
        assert hits[0]["snippet"].startswith("Total net sales")

    @pytest.mark.parametrize("strategy", ["bm25", "dense", "hybrid"])
    def test_strategy_switch(self, client: TestClient, strategy: str) -> None:
        response = client.post(
            "/v1/search", json={"query": "goodwill impairment", "strategy": strategy, "k": 5}
        )
        assert response.status_code == 200, response.text
        assert response.json()["hits"]

    def test_ticker_filter_excludes_other_documents(self, client: TestClient) -> None:
        hits = client.post(
            "/v1/search", json={"query": "revenue", "ticker": "othr", "k": 10}
        ).json()["hits"]
        assert hits and {hit["doc_name"] for hit in hits} == {OTHER_DOC}

    def test_form_filter_is_hyphen_tolerant(self, client: TestClient) -> None:
        hits = client.post("/v1/search", json={"query": "revenue", "form": "10k"}).json()["hits"]
        assert hits
        none = client.post("/v1/search", json={"query": "revenue", "form": "10-Q"}).json()["hits"]
        assert none == []

    def test_bad_strategy_422(self, client: TestClient) -> None:
        assert_problem(client.post("/v1/search", json={"query": "x", "strategy": "faiss"}), 422)

    def test_concurrent_requests_keep_their_own_filters(self, client: TestClient) -> None:
        """Sync endpoints share one DuckDB store across the threadpool; each request must see
        only its own result set (a shared connection leaks rows between threads or trips a
        spurious 503 when one thread consumes another's ``_fts_index_exists`` row)."""
        n_workers, per_worker = 8, 20
        docs = (TOP_DOC, OTHER_DOC)

        def worker(index: int) -> list[str]:
            doc_name = docs[index % len(docs)]
            problems: list[str] = []
            for _ in range(per_worker):
                response = client.post(
                    "/v1/search",
                    json={"query": "revenue income", "doc_names": [doc_name], "k": 10},
                )
                if response.status_code != 200:
                    problems.append(f"{doc_name}: {response.status_code} {response.text}")
                    break
                seen = {hit["doc_name"] for hit in response.json()["hits"]}
                if seen != {doc_name}:
                    problems.append(f"{doc_name}: hits from {sorted(seen)}")
                    break
            return problems

        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            results = list(pool.map(worker, range(n_workers)))
        assert [p for problems in results for p in problems] == []


class TestFilings:
    def test_list_all(self, client: TestClient) -> None:
        docs = client.get("/v1/filings").json()
        assert [doc["doc_name"] for doc in docs] == [TOP_DOC, OTHER_DOC]
        assert docs[0]["ticker"] == TICKER and docs[0]["n_pages"] == 10

    def test_filters(self, client: TestClient) -> None:
        assert [d["doc_name"] for d in client.get("/v1/filings?ticker=fixt").json()] == [TOP_DOC]
        assert [d["doc_name"] for d in client.get("/v1/filings?fiscal_year=2022").json()] == [
            OTHER_DOC
        ]
        assert len(client.get("/v1/filings?form=10K").json()) == 2
        assert client.get("/v1/filings?form=10-Q").json() == []
        assert_problem(client.get("/v1/filings?fiscal_year=abc"), 422)

    def test_page_text(self, client: TestClient) -> None:
        response = client.get(f"/v1/filings/{TOP_DOC}/pages/1")
        assert response.status_code == 200
        assert response.json() == {
            "doc_name": TOP_DOC,
            "page_num": 1,
            "text": NET_SALES_SENTENCE + " Growth was broad-based.",
        }

    def test_page_404s(self, client: TestClient) -> None:
        body = assert_problem(client.get(f"/v1/filings/{TOP_DOC}/pages/99"), 404)
        assert "no page 99" in body["detail"]
        body = assert_problem(client.get("/v1/filings/NOPE_2020_10K/pages/1"), 404)
        assert "not in the index" in body["detail"]
        assert_problem(client.get(f"/v1/filings/{TOP_DOC}/pages/0"), 422)


class TestXbrlQuery:
    def test_select_returns_rows_with_accession_numbers(self, client: TestClient) -> None:
        response = client.post(
            "/v1/xbrl/query",
            json={
                "sql": "SELECT ticker, fy, val, accn FROM xbrl_facts "
                "WHERE fp = 'FY' ORDER BY fy, tag LIMIT 3"
            },
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["columns"] == ["ticker", "fy", "val", "accn"]
        assert body["row_count"] == len(body["rows"]) == 3
        assert body["truncated"] is False
        assert body["rows"][0][0] == TICKER
        assert "LIMIT 3" in body["sql"]

    def test_financials_view_is_queryable(self, client: TestClient) -> None:
        response = client.post(
            "/v1/xbrl/query", json={"sql": "SELECT ticker, fiscal_year FROM financials"}
        )
        assert response.status_code == 200, response.text
        assert response.json()["rows"]

    @pytest.mark.parametrize(
        "sql",
        [
            "DROP TABLE xbrl_facts",
            "SELECT 1; SELECT 2",
            "SELECT * FROM chunks",
            "SELECT * FROM read_csv('/etc/passwd')",
            "INSERT INTO xbrl_facts VALUES (1)",
        ],
    )
    def test_guard_refusals_are_400(self, client: TestClient, sql: str) -> None:
        body = assert_problem(client.post("/v1/xbrl/query", json={"sql": sql}), 400)
        assert body["title"] == "SQL rejected"

    def test_binder_errors_are_400_not_500(self, client: TestClient) -> None:
        body = assert_problem(
            client.post("/v1/xbrl/query", json={"sql": "SELECT no_such_column FROM xbrl_facts"}),
            400,
        )
        assert "no_such_column" in body["detail"]

    def test_blank_sql_422(self, client: TestClient) -> None:
        assert_problem(client.post("/v1/xbrl/query", json={"sql": "   "}), 422)


# ---- OpenAPI ---------------------------------------------------------------------------------


def test_openapi_snapshot(client: TestClient) -> None:
    schema = client.get("/openapi.json").json()
    operations = sorted(
        f"{method.upper()} {path}"
        for path, methods in schema["paths"].items()
        for method in methods
    )
    assert operations == [
        "GET /healthz",
        "GET /readyz",
        "GET /v1/filings",
        "GET /v1/filings/{doc_name}/pages/{page}",
        "GET /version",
        "POST /v1/ask",
        "POST /v1/search",
        "POST /v1/xbrl/query",
    ]
    components = schema["components"]["schemas"]
    for name in (
        "AskRequest",
        "AskResponse",
        "SearchRequest",
        "SearchResponse",
        "PageResponse",
        "SqlRequest",
        "SqlResult",
        "ProblemDetail",
        "ReadyResponse",
        "VersionResponse",
        "DocumentMeta",
        "HitView",
        "Citation",
    ):
        assert name in components, name
    ask = schema["paths"]["/v1/ask"]["post"]
    assert set(ask["responses"]) == {"200", "402", "403", "422", "429", "502", "503", "504"}
    assert components["AskRequest"]["properties"]["mode"]["default"] == "rag"
    assert components["AskRequest"]["properties"]["k"]["maximum"] == 20
    assert components["AskRequest"]["additionalProperties"] is False
    assert client.get("/docs").status_code == 200
