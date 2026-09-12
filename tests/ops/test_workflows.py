"""GitHub Actions workflows (SPEC section 10): shape, guards, smoke steps, no inline secrets."""

from __future__ import annotations

import re
from typing import Any

import pytest

from tests.ops.conftest import WORKFLOWS, load_workflow

EXPECTED = ("ci", "build", "deploy-cloudrun", "deploy-azure", "eval-full")
SECRET_LIKE = re.compile(r"(sk-[A-Za-z0-9]{20,}|sk-ant-[A-Za-z0-9-]{20,}|AKIA[0-9A-Z]{16})")


def _steps(wf: dict[str, Any], job: str) -> list[dict[str, Any]]:
    return list(wf["jobs"][job]["steps"])


def _run_text(wf: dict[str, Any], job: str) -> str:
    return "\n".join(str(s.get("run", "")) for s in _steps(wf, job))


@pytest.mark.parametrize("name", EXPECTED)
def test_workflow_parses_and_pins_actions(name: str) -> None:
    wf = load_workflow(name)
    assert wf["name"] == name
    assert "jobs" in wf and wf["jobs"]
    for job in wf["jobs"].values():
        assert "timeout-minutes" in job, "every job needs a timeout"
        for step in job["steps"]:
            uses = step.get("uses")
            if uses:
                assert re.search(r"@v?\d", uses), f"unpinned action {uses}"
    text = (WORKFLOWS / f"{name}.yml").read_text(encoding="utf-8")
    assert not SECRET_LIKE.search(text), "secret-looking literal in workflow"


def test_only_the_five_workflows_exist() -> None:
    names = sorted(p.stem for p in WORKFLOWS.glob("*.yml"))
    assert names == sorted(EXPECTED)


class TestCI:
    def test_triggers_on_pr_and_main(self) -> None:
        wf = load_workflow("ci")
        assert "pull_request" in wf["on"]
        assert wf["on"]["push"]["branches"] == ["main"]

    def test_test_job_runs_lint_tests_and_mock_smoke(self) -> None:
        wf = load_workflow("ci")
        run = _run_text(wf, "test")
        assert "uv sync --extra dev --extra openai --extra anthropic" in run
        assert "ruff check ." in run and "ruff format --check ." in run
        assert "pytest -q --cov=secqa" in run
        assert "secqa eval --config configs/rag_mock.yaml --limit 6" in run
        assert "--extra local" not in run, "CI must not download model weights"

    def test_docker_job_builds_without_pushing_and_smokes_the_container(self) -> None:
        wf = load_workflow("ci")
        job = wf["jobs"]["docker"]
        assert job["needs"] == "test"
        build = next(
            s for s in job["steps"] if str(s.get("uses", "")).startswith("docker/build-push-action")
        )
        assert build["with"]["push"] is False
        assert build["with"]["load"] is True
        assert build["with"]["target"] == "slim"
        run = _run_text(wf, "docker")
        assert "docker run -d --rm -p 8080:8080" in run
        assert "/healthz" in run and "/readyz" in run and "/v1/ask" in run
        assert '"provider": "mock"' in run
        assert "GITHUB_STEP_SUMMARY" in run

    def test_gitleaks_job_present(self) -> None:
        wf = load_workflow("ci")
        uses = [s.get("uses", "") for s in _steps(wf, "gitleaks")]
        assert any(u.startswith("gitleaks/gitleaks-action@") for u in uses)


class TestBuild:
    def test_publishes_both_targets_to_ghcr(self) -> None:
        wf = load_workflow("build")
        assert wf["on"]["push"]["branches"] == ["main"]
        assert wf["on"]["push"]["tags"] == ["v*"]
        assert wf["permissions"]["packages"] == "write"
        job = wf["jobs"]["publish"]
        assert job["strategy"]["matrix"]["target"] == ["full", "slim"]
        push = next(
            s for s in job["steps"] if str(s.get("uses", "")).startswith("docker/build-push-action")
        )
        assert push["with"]["push"] is True
        assert push["with"]["target"] == "${{ matrix.target }}"
        assert "ghcr.io" in wf["env"]["REGISTRY"]


class TestDeploy:
    @pytest.mark.parametrize(
        ("name", "guard"),
        [
            ("deploy-cloudrun", "vars.DEPLOY_GCP == 'true'"),
            ("deploy-azure", "vars.DEPLOY_AZURE == 'true'"),
        ],
    )
    def test_guarded_and_oidc(self, name: str, guard: str) -> None:
        wf = load_workflow(name)
        job = wf["jobs"]["deploy"]
        assert job["if"] == guard
        assert wf["permissions"]["id-token"] == "write"
        assert wf["on"]["push"]["tags"] == ["v*"]
        assert "workflow_dispatch" in wf["on"]

    @pytest.mark.parametrize("name", ["deploy-cloudrun", "deploy-azure"])
    def test_ends_with_readyz_and_mock_ask_smoke_in_job_summary(self, name: str) -> None:
        wf = load_workflow(name)
        last = _steps(wf, "deploy")[-1]
        run = str(last["run"])
        assert "/readyz" in run and "/v1/ask" in run
        assert '"provider": "mock"' in run
        assert "GITHUB_STEP_SUMMARY" in run
        assert 'a["citations"]' in run, "smoke asserts the mock answer carries citations"

    def test_cloudrun_flags_match_spec(self) -> None:
        wf = load_workflow("deploy-cloudrun")
        run = _run_text(wf, "deploy")
        for flag in (
            "--memory 2Gi",
            "--cpu 1",
            "--cpu-boost",
            "--max-instances 2",
            "--concurrency 4",
            "--timeout 120",
            "--set-secrets",
            "--set-env-vars",
        ):
            assert flag in run, flag
        assert "us-central1" in str(wf["jobs"]["deploy"]["env"]["REGION"])
        assert "gcloud run deploy" in run
        assert "docker.pkg.dev" in run, (
            "Cloud Run pulls from Artifact Registry, so the image is mirrored"
        )
        uses = [s.get("uses", "") for s in _steps(wf, "deploy")]
        assert any(u.startswith("google-github-actions/auth@") for u in uses)

    def test_azure_uses_bicep_template(self) -> None:
        wf = load_workflow("deploy-azure")
        steps = _steps(wf, "deploy")
        uses = [s.get("uses", "") for s in steps]
        assert any(u.startswith("azure/login@") for u in uses)
        script = "\n".join(str(s.get("with", {}).get("inlineScript", "")) for s in steps)
        assert "az deployment group create" in script
        assert "infra/azure/main.bicep" in script
        assert 'openaiApiKey="${{ secrets.OPENAI_API_KEY }}"' in script


class TestEvalFull:
    def test_dispatch_only_and_commits_results_via_pr(self) -> None:
        wf = load_workflow("eval-full")
        assert list(wf["on"]) == ["workflow_dispatch"]
        run = _run_text(wf, "eval")
        assert "--extra local" in run
        assert "secqa eval --config" in run
        assert "secqa doctor" in run
        pr = next(
            s
            for s in _steps(wf, "eval")
            if str(s.get("uses", "")).startswith("peter-evans/create-pull-request")
        )
        paths = pr["with"]["add-paths"]
        assert "predictions.jsonl" in paths and "summary.json" in paths and "config.json" in paths
        assert "cassettes" not in paths, "cassettes are Release assets, never committed"
        upload = next(
            s
            for s in _steps(wf, "eval")
            if str(s.get("uses", "")).startswith("actions/upload-artifact")
        )
        assert "artifacts/*.tar.zst" in upload["with"]["path"]
