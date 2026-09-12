"""Dockerfile, .dockerignore, docker-compose.yml and entrypoint.sh (static checks + a real run of
the entrypoint's bootstrap path). No docker daemon is needed."""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from tests.ops.conftest import FIXTURES, REPO, SCRIPTS

DOCKERFILE = REPO / "Dockerfile"
DOCKERIGNORE = REPO / ".dockerignore"
COMPOSE = REPO / "docker-compose.yml"
ENTRYPOINT = SCRIPTS / "entrypoint.sh"

SECRET_ENV_NAMES = ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "SECQA_API_KEY", "SEC_USER_AGENT")


def _stages(text: str) -> dict[str, str]:
    """``{stage name: base}`` from every ``FROM <base> AS <name>`` line."""
    return {
        m.group("name"): m.group("base")
        for m in re.finditer(r"^FROM\s+(?P<base>\S+)\s+AS\s+(?P<name>\S+)", text, re.M | re.I)
    }


def _stage_body(text: str, name: str) -> str:
    """Instructions of one stage (from its FROM line up to the next FROM)."""
    pattern = re.compile(rf"^FROM\s+\S+\s+AS\s+{re.escape(name)}\s*$", re.M | re.I)
    match = pattern.search(text)
    assert match is not None, f"stage {name} not found"
    rest = text[match.end() :]
    nxt = re.search(r"^FROM\s", rest, re.M)
    return rest if nxt is None else rest[: nxt.start()]


@pytest.fixture(scope="module")
def dockerfile() -> str:
    return DOCKERFILE.read_text(encoding="utf-8")


class TestDockerfile:
    def test_multi_stage_with_full_and_slim_targets(self, dockerfile: str) -> None:
        stages = _stages(dockerfile)
        assert {"deps", "build-slim", "build-full", "runtime", "slim", "full"} <= set(stages)
        assert stages["deps"] == "${UV_IMAGE}"
        assert "ghcr.io/astral-sh/uv:0.11-python3.11-bookworm-slim" in dockerfile
        assert stages["runtime"] == "${PYTHON_IMAGE}"
        assert "python:3.11-slim" in dockerfile
        assert stages["slim"] == "runtime" and stages["full"] == "runtime"

    def test_extras_per_target(self, dockerfile: str) -> None:
        slim = _stage_body(dockerfile, "build-slim")
        full = _stage_body(dockerfile, "build-full")
        assert "uv sync --frozen --no-dev --extra api-embeddings" in slim
        assert "--extra local" not in slim
        assert "uv sync --frozen --no-dev --extra local" in full
        assert "SentenceTransformer('BAAI/bge-small-en-v1.5'" in full  # weights baked at build
        assert "SentenceTransformer" not in slim

    def test_caches_warmed_at_build(self, dockerfile: str) -> None:
        for stage in ("build-slim", "build-full"):
            body = _stage_body(dockerfile, stage)
            assert "INSTALL fts" in body and "LOAD fts" in body
            assert "tiktoken.get_encoding('cl100k_base')" in body
        for stage in ("slim", "full"):
            body = _stage_body(dockerfile, stage)
            assert "/home/app/.duckdb" in body  # extension dir copied to the runtime user's HOME
            assert "/opt/caches" in body
        assert "TIKTOKEN_CACHE_DIR=/opt/caches/tiktoken" in dockerfile
        assert "HF_HOME=/opt/caches/hf" in dockerfile
        assert "HF_HUB_OFFLINE=1" in _stage_body(dockerfile, "full")

    def test_runtime_is_non_root_with_healthcheck_and_entrypoint(self, dockerfile: str) -> None:
        runtime = _stage_body(dockerfile, "runtime")
        assert re.search(r"^USER app\s*$", runtime, re.M)
        assert runtime.index("USER app") > runtime.index("COPY --chown=app:app")
        assert re.search(r"^HEALTHCHECK\b", runtime, re.M)
        assert "/healthz" in runtime
        assert re.search(r'^ENTRYPOINT \["/app/scripts/entrypoint.sh"\]', runtime, re.M)
        assert re.search(r"^EXPOSE 8080", runtime, re.M)
        assert "SECQA_DUCKDB_PATH=/data/index.duckdb" in runtime
        assert "SECQA_GIT_SHA=${GIT_SHA}" in runtime

    def test_no_secrets_baked_in(self, dockerfile: str) -> None:
        for name in SECRET_ENV_NAMES:
            assert not re.search(rf"^\s*(ENV|ARG)\s+{name}\b", dockerfile, re.M), name
        assert "sk-" not in dockerfile

    def test_uses_frozen_lockfile_and_no_dev(self, dockerfile: str) -> None:
        instructions = "\n".join(
            ln for ln in dockerfile.splitlines() if not ln.lstrip().startswith("#")
        )
        syncs = re.findall(r"uv sync [^\\\n]+", instructions)
        assert syncs, "no uv sync calls"
        for call in syncs:
            assert "--frozen" in call and "--no-dev" in call, call
        # The runtime stage copies /opt/venv alone, so every sync that installs the project
        # must install it as a regular package (an editable .pth would point at /build/src).
        project_syncs = [c for c in syncs if "--no-install-project" not in c]
        assert project_syncs, "no uv sync installs the project"
        for call in project_syncs:
            assert "--no-editable" in call, call


class TestDockerignore:
    def test_excludes_secrets_data_and_vcs(self) -> None:
        lines = {
            ln.strip()
            for ln in DOCKERIGNORE.read_text(encoding="utf-8").splitlines()
            if ln.strip() and not ln.startswith("#")
        }
        for pattern in (".git", ".venv", ".env", "data/*", "results", "cassettes", "*.pdf"):
            assert pattern in lines, pattern
        # Runtime inputs the Dockerfile COPYs must be allow-listed back in.
        for keep in (
            "!data/companies.yaml",
            "!tests/fixtures/eval_fixture_pages.json",
            "!tests/fixtures/fb_mini.jsonl",
            "!scripts/entrypoint.sh",
            "!scripts/bootstrap_demo_index.py",
            "!README.md",
        ):
            assert keep in lines, keep

    def test_every_copied_path_is_not_ignored(self, dockerfile: str) -> None:
        """Each source the Dockerfile COPYs from the context exists in the repo."""
        copies = re.findall(r"^COPY(?: --chown=\S+)?\s+(.+?)\s+\S+\s*$", dockerfile, re.M)
        sources = [src for line in copies if "--from=" not in line for src in line.split()]
        assert sources
        for src in sources:
            assert (REPO / src).exists(), src


class TestCompose:
    def test_compose_service_shape(self) -> None:
        compose = yaml.safe_load(COMPOSE.read_text(encoding="utf-8"))
        api = compose["services"]["api"]
        assert api["build"]["dockerfile"] == "Dockerfile"
        assert "${IMAGE_TARGET:-full}" in api["build"]["target"]
        assert any(str(p).endswith(":8080") for p in api["ports"])
        assert api["environment"]["SECQA_DUCKDB_PATH"] == "/data/index.duckdb"
        assert "secqa-data" in compose["volumes"]
        assert not any(name in api["environment"] for name in SECRET_ENV_NAMES), (
            "keys come from .env, never from the compose file"
        )
        env_files = api["env_file"]
        assert env_files[0]["path"] == ".env" and env_files[0]["required"] is False


class TestEntrypoint:
    def test_is_executable_posix_sh(self) -> None:
        text = ENTRYPOINT.read_text(encoding="utf-8")
        assert text.startswith("#!/bin/sh")
        assert "set -eu" in text
        assert os.access(ENTRYPOINT, os.X_OK)
        assert "exec uvicorn secqa.api.app:app" in text
        assert '--port "${PORT}"' in text
        assert 'exec "$@"' in text

    def test_execs_given_command_when_bootstrap_skipped(self, tmp_path: Path) -> None:
        env = {**os.environ, "SECQA_SKIP_BOOTSTRAP": "1", "SECQA_DUCKDB_PATH": str(tmp_path / "x")}
        result = subprocess.run(
            ["sh", str(ENTRYPOINT), "echo", "hello-from-entrypoint"],
            capture_output=True,
            text=True,
            env=env,
            timeout=30,
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "hello-from-entrypoint"
        assert not (tmp_path / "x").exists()

    def test_bootstrap_builds_fixture_index_then_execs(self, tmp_path: Path) -> None:
        """The container's zero-config path: no SECQA_INDEX_URL -> fixture index is built."""
        duckdb_path = tmp_path / "data" / "index.duckdb"
        env = {
            **os.environ,
            "PATH": str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", ""),
            "SECQA_APP_DIR": str(REPO),
            "SECQA_FIXTURE_PAGES": str(FIXTURES / "eval_fixture_pages.json"),
            "SECQA_DUCKDB_PATH": str(duckdb_path),
            "SECQA_EMBEDDER": "hashing",
            "SECQA_LOG_JSON": "true",
        }
        env.pop("SECQA_INDEX_URL", None)
        result = subprocess.run(
            ["sh", str(ENTRYPOINT), "true"],
            capture_output=True,
            text=True,
            env=env,
            timeout=120,
        )
        assert result.returncode == 0, result.stderr
        assert duckdb_path.is_file()
        assert '"event": "fixture_index_ready"' in result.stderr
