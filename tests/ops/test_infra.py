"""infra/ (Cloud Run env example, Azure Bicep), Makefile, pre-commit and gitleaks configs."""

from __future__ import annotations

import re
import tomllib

import yaml

from tests.ops.conftest import REPO

BICEP = REPO / "infra" / "azure" / "main.bicep"
CLOUDRUN_ENV = REPO / "infra" / "gcp" / "cloudrun.env.example"
MAKEFILE = REPO / "Makefile"
PRE_COMMIT = REPO / ".pre-commit-config.yaml"
GITLEAKS = REPO / ".gitleaks.toml"


class TestAzureBicep:
    def test_shape_matches_spec(self) -> None:
        text = BICEP.read_text(encoding="utf-8")
        assert "targetScope = 'resourceGroup'" in text
        assert "Microsoft.App/containerApps@" in text
        assert "Microsoft.App/managedEnvironments@" in text
        assert "param minReplicas int = 0" in text
        assert "param maxReplicas int = 2" in text
        assert "param cpu string = '1.0'" in text
        assert "param memory string = '2Gi'" in text
        assert "var containerPort = 8080" in text
        assert "external: true" in text
        assert "output fqdn string" in text

    def test_keys_are_secure_parameters_never_literals(self) -> None:
        text = BICEP.read_text(encoding="utf-8")
        for param in ("openaiApiKey", "anthropicApiKey", "secqaApiKey"):
            match = re.search(
                rf"@secure\(\)\s*\n@description\(.*\)\s*\nparam {param} string = ''", text
            )
            assert match, f"{param} must be an empty-default @secure() parameter"
        assert "secretRef: 'openai-api-key'" in text
        assert not re.search(r"sk-[A-Za-z0-9]{10,}", text)

    def test_probes_hit_health_and_ready(self) -> None:
        text = BICEP.read_text(encoding="utf-8")
        assert "path: '/healthz'" in text and "path: '/readyz'" in text

    def test_readme_present_with_status_wording(self) -> None:
        readme = (REPO / "infra" / "azure" / "README.md").read_text(encoding="utf-8")
        assert "DEPLOY_AZURE" in readme
        assert "deployment not yet verified" in readme

    def test_anonymous_ghcr_pull_documents_the_package_visibility_step(self) -> None:
        """The template passes no ``configuration.registries`` credential, so Container Apps
        pulls from GHCR anonymously. A package first pushed by ``build.yml`` with
        ``GITHUB_TOKEN`` is private by default, so the one-time "make the package public" step
        must be written down next to every claim that no registry credential is needed; the
        ``full`` and ``-slim`` tags share one package, which the step must say."""
        bicep = BICEP.read_text(encoding="utf-8")
        assert "registries" not in bicep.split("resource app ")[1], (
            "the app pulls anonymously; if a registry credential is added, drop this test"
        )
        assert "public" in bicep and "README.md" in bicep, "the header points at the setup step"
        readme = (REPO / "infra" / "azure" / "README.md").read_text(encoding="utf-8")
        assert "Change visibility" in readme and "Public" in readme
        assert "GITHUB_TOKEN" in readme and "private" in readme, "says why the step exists"
        assert "-slim" in readme, "both targets live in the same package"
        assert "no registry credential is required" not in readme, "the old unconditional claim"
        deploy = (REPO / "docs" / "DEPLOY.md").read_text(encoding="utf-8")
        azure = deploy.split("## Azure Container Apps")[1].split("## Publishing the index")[0]
        assert "Change visibility" in azure and "GITHUB_TOKEN" in azure
        build = (REPO / ".github" / "workflows" / "build.yml").read_text(encoding="utf-8")
        assert "private" in build and "infra/azure/README.md" in build


class TestCloudRun:
    def test_env_example_is_yaml_without_secrets(self) -> None:
        raw = yaml.safe_load(CLOUDRUN_ENV.read_text(encoding="utf-8"))
        assert isinstance(raw, dict)
        assert raw["SECQA_DUCKDB_PATH"] == "/data/index.duckdb"
        assert "SECQA_INDEX_URL" in raw and "SECQA_EMBEDDER" in raw
        for key in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "SECQA_API_KEY"):
            assert key not in raw, f"{key} belongs in Secret Manager"
        assert all(isinstance(v, str) for v in raw.values()), "gcloud env files need string values"

    def test_readme_documents_wif_and_secret_manager(self) -> None:
        readme = (REPO / "infra" / "gcp" / "README.md").read_text(encoding="utf-8")
        assert "workload-identity-pools" in readme
        assert "gcloud secrets create" in readme
        assert "--set-secrets" in readme
        assert "DEPLOY_GCP" in readme


class TestMakefile:
    def test_required_targets(self) -> None:
        text = MAKEFILE.read_text(encoding="utf-8")
        targets = set(re.findall(r"^([a-zA-Z_-]+):", text, re.M))
        for target in ("test", "lint", "smoke-eval", "docker-build", "report", "serve", "setup"):
            assert target in targets, target
        assert "docker build --target full" in text
        assert "run --no-sync pytest -q" in text
        assert "secqa report results/ --out RESULTS.md" in text


class TestHooks:
    def test_pre_commit_config(self) -> None:
        cfg = yaml.safe_load(PRE_COMMIT.read_text(encoding="utf-8"))
        repos = {r["repo"]: r for r in cfg["repos"]}
        assert "https://github.com/astral-sh/ruff-pre-commit" in repos
        assert "https://github.com/gitleaks/gitleaks" in repos
        hook_ids = {h["id"] for r in cfg["repos"] for h in r["hooks"]}
        assert {
            "ruff",
            "ruff-format",
            "gitleaks",
            "detect-private-key",
            "no-dataset-or-index-files",
        } <= hook_ids
        local = next(
            h for r in cfg["repos"] for h in r["hooks"] if h["id"] == "no-dataset-or-index-files"
        )
        pattern = re.compile(local["files"])
        for blocked in (
            "data/x.pdf",
            "data/index.duckdb",
            "cassettes/run.tar.zst",
            ".env",
            "sub/.env",
        ):
            assert pattern.search(blocked), blocked
        assert not pattern.search("README.md")
        assert re.compile(local["exclude"]).search(".env.example")

    def test_gitleaks_config_parses_and_extends_default(self) -> None:
        cfg = tomllib.loads(GITLEAKS.read_text(encoding="utf-8"))
        assert cfg["extend"]["useDefault"] is True
        paths = cfg["allowlist"]["paths"]
        assert any(".env" in p for p in paths)
        assert any("tests/fixtures" in p for p in paths)
        for regex in cfg["allowlist"]["regexes"]:
            re.compile(regex)  # every allowlist regex must be valid
