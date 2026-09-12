"""Cassette data-policy statements must match what ``ReplayCacheProvider`` writes.

A cassette entry stores the full request (system prompt and messages), and the answering, oracle
and judge prompts embed FinanceBench text (question, reference answer, justification, gold
evidence pages). Every document that describes cassettes must therefore say that they contain
dataset text and are shared under CC-BY-NC-4.0 with attribution, and no document may claim the
dataset is never redistributed without qualifying that the claim covers git only.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from secqa.core.contracts import Message
from secqa.providers.base import canonical_request

REPO = Path(__file__).resolve().parents[2]

CASSETTE_DOCS: tuple[Path, ...] = (
    REPO / "cassettes" / "README.md",
    REPO / "LIMITATIONS.md",
    REPO / "NOTICE",
    REPO / "README.md",
    REPO / "docs" / "EVAL.md",
    REPO / "docs" / "decisions.md",
)
LICENCE_DOCS: tuple[Path, ...] = (REPO / "README.md", REPO / "NOTICE", REPO / "SPEC.md")


def test_canonical_request_keeps_the_prompt_text() -> None:
    """The stored request carries every message verbatim (nothing is hashed away)."""
    question = "Question: What was FIXTURE CORP's fiscal 2023 revenue?"
    request = canonical_request("mock", "m", [Message(role="user", content=question)], system="s")
    assert request["messages"][0]["content"] == question
    assert request["system"] == "s"


@pytest.mark.parametrize("path", CASSETTE_DOCS, ids=lambda p: str(p.relative_to(REPO)))
def test_cassette_docs_say_cassettes_contain_dataset_text(path: Path) -> None:
    """Every document describing cassettes names the licence and says they carry dataset text."""
    text = path.read_text(encoding="utf-8")
    assert "cassette" in text.lower(), f"{path.name} no longer mentions cassettes"
    assert "CC-BY-NC-4.0" in text, f"{path.name} must name the FinanceBench licence"
    assert "attribution" in text.lower(), f"{path.name} must say cassettes carry attribution"
    assert "dataset text" in text or "text from the dataset" in text, (
        f"{path.name} must state that cassettes contain FinanceBench text"
    )
    # The old, wrong claim: cassettes hold filing passages and 'our' prompts only.
    assert "prompts (ours)" not in text, f"{path.name} understates what cassettes contain"


@pytest.mark.parametrize("path", LICENCE_DOCS, ids=lambda p: str(p.relative_to(REPO)))
def test_no_unqualified_never_redistributed_claim(path: Path) -> None:
    """'never/not redistributed' may only appear qualified as covering git, not Release assets."""
    text = path.read_text(encoding="utf-8")
    lowered = text.lower()
    for claim in ("never redistributed", "not redistributed"):
        start = 0
        while (idx := lowered.find(claim, start)) != -1:
            window = lowered[idx : idx + len(claim) + 12]
            assert window.startswith(f"{claim} in git"), (
                f"{path.name}: {claim!r} must be qualified as 'in git' (cassettes are Release "
                f"assets that contain dataset text): ...{text[idx : idx + 80]!r}"
            )
            start = idx + len(claim)
