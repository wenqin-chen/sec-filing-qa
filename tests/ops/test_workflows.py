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

    @pytest.mark.parametrize(
        ("name", "image_env", "consumer"),
        [
            ("deploy-cloudrun", "SOURCE_IMAGE", "docker pull"),
            ("deploy-azure", "IMAGE", "az deployment group create"),
        ],
    )
    def test_waits_for_the_image_before_using_it(
        self, name: str, image_env: str, consumer: str
    ) -> None:
        """build.yml fires on the same ``v*`` tag push and needs many minutes for the ``full``
        target, so a deploy that references ``:<sha>`` immediately races it. Each deploy job must
        poll the registry for the exact image it will use, before the first step that uses it,
        with a hard deadline that leaves the job time to deploy and smoke."""
        wf = load_workflow(name)
        job = wf["jobs"]["deploy"]
        image = str(job["env"][image_env])
        assert image.startswith("ghcr.io/") and "${{ inputs.image_tag || github.sha }}" in image
        steps = _steps(wf, "deploy")
        runs = [str(s.get("run", "")) for s in steps]
        scripts = [str(s.get("with", {}).get("inlineScript", "")) for s in steps]
        wait_idx = next(
            i for i, run in enumerate(runs) if f'docker manifest inspect "${{{image_env}}}"' in run
        )
        consumer_idx = next(
            i for i, (r, s) in enumerate(zip(runs, scripts, strict=True)) if consumer in r + s
        )
        assert wait_idx < consumer_idx, "the wait must precede the first use of the image"
        wait = steps[wait_idx]
        minutes = int(wait["env"]["IMAGE_WAIT_MINUTES"])
        assert 20 <= minutes <= 55, "long enough for the torch build, short of the job timeout"
        assert "DEADLINE" in runs[wait_idx] and "exit 1" in runs[wait_idx], "bounded, fails loudly"
        assert "::error::" in runs[wait_idx] and "build.yml" in runs[wait_idx]
        assert int(job["timeout-minutes"]) >= minutes + 10, "room for the deploy + smoke"
        logins = [
            i
            for i, s in enumerate(steps)
            if str(s.get("uses", "")).startswith("docker/login-action")
            and s["with"]["registry"] == "ghcr.io"
        ]
        assert logins and logins[0] < wait_idx, "GHCR login precedes the manifest check"
        assert wf["permissions"]["packages"] == "read"

    def test_azure_checks_anonymous_pull_between_the_wait_and_the_deploy(self) -> None:
        """The Bicep passes no registry credential, so Container Apps pulls the GHCR image
        anonymously, and a package first pushed with ``GITHUB_TOKEN`` is private by default. The
        wait step's ``docker manifest inspect`` runs under the GHCR login and therefore passes on
        a private package; a separate unauthenticated manifest fetch (token endpoint with the
        pull scope, then ``/v2/<repo>/manifests/<tag>``) must sit between the wait and the
        deploy and fail loudly with the "make the package public" instruction."""
        wf = load_workflow("deploy-azure")
        steps = _steps(wf, "deploy")
        runs = [str(s.get("run", "")) for s in steps]
        scripts = [str(s.get("with", {}).get("inlineScript", "")) for s in steps]
        wait_idx = next(i for i, r in enumerate(runs) if 'docker manifest inspect "${IMAGE}"' in r)
        deploy_idx = next(i for i, s in enumerate(scripts) if "az deployment group create" in s)
        check_idx = next(i for i, r in enumerate(runs) if "https://ghcr.io/token?scope=" in r)
        assert wait_idx < check_idx < deploy_idx, "wait, then anonymous check, then deploy"
        check = runs[check_idx]
        assert "/manifests/" in check and "/v2/" in check, "fetches the manifest of the exact tag"
        assert "${IMAGE" in check, "checks the image the Bicep will deploy"
        assert "secrets.GITHUB_TOKEN" not in check and not re.search(r"^\s*docker ", check, re.M), (
            "must not reuse the authenticated docker login, which hides a private package"
        )
        assert '"200"' in check and "exit 1" in check and "::error::" in check
        assert "Change visibility" in check and "infra/azure/README.md" in check, (
            "the failure names the one-time fix"
        )

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

    def test_run_id_is_chosen_before_the_eval_and_never_discovered_by_sorting(self) -> None:
        """Run ids are ``<git_sha7>_<YYYYMMDD-HHMM>``, so ``ls results/<config>/ | sort`` orders
        by commit hash, not by time, and the checkout already contains every committed run of
        the config. The workflow must decide the id first (with the runner's own
        ``make_run_id``), pass it to ``secqa eval --run-id``, check the files it will publish
        exist, and hand that same id to the later steps."""
        wf = load_workflow("eval-full")
        step = next(s for s in _steps(wf, "eval") if s.get("id") == "run")
        run = str(step["run"])
        assert "| sort" not in run and "ls -" not in run and "tail -n" not in run, (
            "the run directory must not be guessed from a directory listing"
        )
        assert "make_run_id" in run and "${{ github.sha }}" in run
        id_line = run.index("RUN_ID=$(")
        eval_lines = [line for line in run.splitlines() if "secqa eval --config" in line]
        assert len(eval_lines) == 2 and all('--run-id "${RUN_ID}"' in line for line in eval_lines)
        assert id_line < run.index("secqa eval --config"), "id decided before the eval runs"
        assert 'RUN_DIR="results/${CONFIG_NAME}/${RUN_ID}"' in run
        assert 'if [ -e "${RUN_DIR}" ]' in run and "exit 1" in run, "never resumes an old run"
        for name in ("config.json", "predictions.jsonl", "summary.json"):
            assert name in run, f"asserts {name} was written to the published directory"
        assert run.index("summary.json") > run.index("secqa eval --config")
        assert 'echo "run_id=${RUN_ID}"' in run and 'echo "run_dir=${RUN_DIR}"' in run
        later = "\n".join(str(s.get("run", "")) for s in _steps(wf, "eval")[1:])
        assert "steps.run.outputs.run_id" in later
