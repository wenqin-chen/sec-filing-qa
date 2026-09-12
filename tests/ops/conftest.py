"""Fixtures for the ops module tests: repository paths, a loader for ``scripts/*.py`` and a
synthetic :class:`EvalRecord` factory. Everything is offline."""

from __future__ import annotations

import importlib.util
import sys
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import yaml

from secqa.core.contracts import EvalRecord, Usage

REPO = Path(__file__).resolve().parents[2]
SCRIPTS = REPO / "scripts"
WORKFLOWS = REPO / ".github" / "workflows"
FIXTURES = REPO / "tests" / "fixtures"


def load_script(name: str) -> ModuleType:
    """Import ``scripts/<name>.py`` as a module (registered in ``sys.modules`` so dataclasses
    and ``__module__`` lookups behave like a normal import)."""
    path = SCRIPTS / f"{name}.py"
    module_name = f"secqa_scripts.{name}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None, path
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def load_workflow(name: str) -> dict[str, Any]:
    """Parse ``.github/workflows/<name>.yml``; the ``on`` key (PyYAML reads it as ``True``) is
    normalised to the string ``'on'``."""
    raw = yaml.safe_load((WORKFLOWS / f"{name}.yml").read_text(encoding="utf-8"))
    assert isinstance(raw, dict), name
    if True in raw:
        raw["on"] = raw.pop(True)
    return raw


@pytest.fixture(scope="session")
def repo() -> Path:
    return REPO


@pytest.fixture
def script() -> Callable[[str], ModuleType]:
    return load_script


@pytest.fixture
def eval_record() -> Callable[..., EvalRecord]:
    """Factory for a minimal, valid :class:`EvalRecord` (no dataset text)."""

    def _make(**overrides: Any) -> EvalRecord:
        base: dict[str, Any] = {
            "financebench_id": "fb_synthetic_001",
            "question_type": "metrics-generated",
            "config_name": "rag_mock",
            "run_id": "abc1234_20260911-1200",
            "git_sha": "abc1234",
            "index_sha": "deadbeef",
            "prompt_hashes": {},
            "models_yaml_as_of": "2026-09-11",
            "provider": "mock",
            "model": "mock",
            "mode": "rag",
            "embedder": "hashing-384",
            "strategy": "hybrid",
            "k": 4,
            "answer_text": "Total net sales were $1,577 million.",
            "value": 1577.0,
            "unit": "million USD",
            "abstained": False,
            "grounded": True,
            "citations": [],
            "retrieved_pages": [["FIXTURE_2023_10K", 1]],
            "gold_pages": [["FIXTURE_2023_10K", 1]],
            "page_recall_5": 1.0,
            "page_recall_10": 1.0,
            "page_recall_20": 1.0,
            "overlap_recall_10": 1.0,
            "gold_page_mrr": 1.0,
            "numeric_match": True,
            "judge": None,
            "faith": None,
            "citation_verified_rate": 1.0,
            "failure": "none",
            "usage": Usage(),
            "cost_usd": 0.0,
            "judge_cost_usd": 0.0,
            "latency_ms": 12.0,
            "retrieval_ms": 5.0,
            "llm_ms": 7.0,
            "steps": 1,
            "tool_calls": 0,
            "terminated_by": "single_shot",
            "error": None,
            "timestamp": datetime(2026, 9, 11, 12, 0, tzinfo=UTC),
        }
        base.update(overrides)
        return EvalRecord.model_validate(base)

    return _make
