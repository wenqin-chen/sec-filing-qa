"""get_embedder: spec grammar, key handling and error messages."""

from __future__ import annotations

import sys
from collections.abc import Callable

import pytest

from secqa.core.errors import ConfigError
from secqa.core.settings import Settings
from secqa.embeddings import HashingEmbedder, OpenAIEmbedder, get_embedder, parse_spec


def test_hashing_default() -> None:
    embedder = get_embedder("hashing")
    assert isinstance(embedder, HashingEmbedder)
    assert embedder.dim == 384
    assert embedder.name == "hashing-384"


def test_hashing_with_dim_and_whitespace_case() -> None:
    embedder = get_embedder("  Hashing:128 ")
    assert isinstance(embedder, HashingEmbedder)
    assert embedder.dim == 128


@pytest.mark.parametrize("spec", ["hashing:abc", "hashing:0", "hashing:-1"])
def test_hashing_bad_dim(spec: str) -> None:
    with pytest.raises(ConfigError):
        get_embedder(spec)


@pytest.mark.parametrize("spec", ["", "   ", "anthropic", "bge-small", "openai-embeddings"])
def test_unknown_spec(spec: str) -> None:
    with pytest.raises(ConfigError, match="embedder"):
        get_embedder(spec)


def test_parse_spec() -> None:
    assert parse_spec("openai:text-embedding-3-large") == ("openai", "text-embedding-3-large")
    assert parse_spec("local") == ("local", "")
    with pytest.raises(ConfigError):
        parse_spec("nope:x")


def test_openai_without_key_is_config_error() -> None:
    with pytest.raises(ConfigError, match="OPENAI_API_KEY"):
        get_embedder("openai")


def test_openai_with_explicit_settings(settings_override: Callable[..., Settings]) -> None:
    settings = settings_override(openai_api_key="sk-test-not-a-real-key")
    embedder = get_embedder("openai", settings)
    assert isinstance(embedder, OpenAIEmbedder)
    assert embedder.name == "text-embedding-3-small"
    assert embedder.dim == 384
    embedder.close()


def test_openai_model_override_and_process_settings(
    settings_override: Callable[..., Settings],
) -> None:
    settings_override(openai_api_key="sk-test-not-a-real-key")  # exports OPENAI_API_KEY
    embedder = get_embedder("openai:text-embedding-3-large")
    assert isinstance(embedder, OpenAIEmbedder)
    assert embedder.model == "text-embedding-3-large"
    embedder.close()


def test_local_without_sentence_transformers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)  # forces ImportError
    with pytest.raises(ConfigError, match="extra local"):
        get_embedder("local")
