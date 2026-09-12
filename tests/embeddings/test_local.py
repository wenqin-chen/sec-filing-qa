"""LocalEmbedder: model-card prefixes and wiring (offline via a stub), real model (slow)."""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from secqa.core.contracts import Embedder
from secqa.core.errors import ConfigError
from secqa.embeddings.local import DEFAULT_LOCAL_MODEL, LocalEmbedder, prefixes_for

BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class _StubSentenceTransformer:
    """Records constructor args and inputs; returns a fixed-width bag-of-characters vector."""

    instances: list[_StubSentenceTransformer] = []

    def __init__(self, model_name: str, **kwargs: Any) -> None:
        self.model_name = model_name
        self.kwargs = kwargs
        self.encoded: list[list[str]] = []
        self.encode_kwargs: list[dict[str, Any]] = []
        _StubSentenceTransformer.instances.append(self)

    def get_sentence_embedding_dimension(self) -> int:
        return 8

    def encode(self, sentences: list[str], **kwargs: Any) -> np.ndarray:
        self.encoded.append(list(sentences))
        self.encode_kwargs.append(kwargs)
        out = np.zeros((len(sentences), 8), dtype=np.float32)
        for row, text in enumerate(sentences):
            for char in text.encode():
                out[row, char % 8] += 1.0
        return out


@pytest.fixture
def stub_sentence_transformers(monkeypatch: pytest.MonkeyPatch) -> type[_StubSentenceTransformer]:
    module = types.ModuleType("sentence_transformers")
    module.SentenceTransformer = _StubSentenceTransformer  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    _StubSentenceTransformer.instances.clear()
    return _StubSentenceTransformer


def test_prefixes_per_model_card() -> None:
    assert prefixes_for("BAAI/bge-small-en-v1.5") == (BGE_QUERY_PREFIX, "")
    assert prefixes_for("BAAI/bge-base-en") == (BGE_QUERY_PREFIX, "")
    assert prefixes_for("/models/bge-large-en-v1.5") == (BGE_QUERY_PREFIX, "")
    assert prefixes_for("BAAI/bge-m3") == ("", "")
    assert prefixes_for("intfloat/e5-small-v2") == ("query: ", "passage: ")
    assert prefixes_for("intfloat/multilingual-e5-base") == ("query: ", "passage: ")
    assert prefixes_for("sentence-transformers/all-MiniLM-L6-v2") == ("", "")


def test_wiring_with_stub(
    stub_sentence_transformers: type[_StubSentenceTransformer], tmp_path: Path
) -> None:
    embedder = LocalEmbedder(cache_dir=tmp_path / "hf")
    assert isinstance(embedder, Embedder)
    assert embedder.name == "bge-small-en-v1.5"
    assert embedder.model_name == DEFAULT_LOCAL_MODEL
    assert embedder.dim == 8
    (model,) = stub_sentence_transformers.instances
    assert model.model_name == DEFAULT_LOCAL_MODEL
    assert model.kwargs == {"device": "cpu", "cache_folder": str(tmp_path / "hf")}

    passages = embedder.embed(["net sales", "", "operating income"], batch_size=1)
    queries = embedder.embed(["net sales"], kind="query")
    assert passages.shape == (3, 8) and queries.shape == (1, 8)
    assert passages.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(passages[[0, 2]], axis=1), 1.0, atol=1e-6)
    assert not passages[1].any()  # blank text never reaches the model
    # Prefix only on queries, batching honoured, sentence-transformers normalisation requested.
    assert model.encoded == [["net sales"], ["operating income"], [BGE_QUERY_PREFIX + "net sales"]]
    assert all(kw["normalize_embeddings"] is True for kw in model.encode_kwargs)
    assert not np.allclose(passages[0], queries[0])


def test_no_prefix_for_unknown_model(
    stub_sentence_transformers: type[_StubSentenceTransformer],
) -> None:
    embedder = LocalEmbedder(model_name="sentence-transformers/all-MiniLM-L6-v2", device="mps")
    assert embedder.name == "all-MiniLM-L6-v2"
    embedder.embed(["net sales"], kind="query")
    (model,) = stub_sentence_transformers.instances
    assert model.kwargs == {"device": "mps"}
    assert model.encoded == [["net sales"]]


def test_empty_model_name_rejected(
    stub_sentence_transformers: type[_StubSentenceTransformer],
) -> None:
    with pytest.raises(ConfigError):
        LocalEmbedder(model_name="  ")


def test_missing_dependency_is_config_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    with pytest.raises(ConfigError, match="sentence-transformers"):
        LocalEmbedder()


@pytest.mark.slow
def test_real_bge_small_downloads_and_embeds() -> None:
    """Opt-in (``-m slow``): downloads BAAI/bge-small-en-v1.5 and checks the real geometry."""
    pytest.importorskip("sentence_transformers")
    embedder = LocalEmbedder()
    assert embedder.name == "bge-small-en-v1.5"
    assert embedder.dim == 384
    passages = embedder.embed(
        [
            "Total net sales were $1,577 million in fiscal 2023.",
            "The board approved a new share repurchase programme.",
        ]
    )
    query = embedder.embed(["What were net sales in fiscal 2023?"], kind="query")
    assert passages.shape == (2, 384) and passages.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(passages, axis=1), 1.0, atol=1e-5)
    sims = passages @ query[0]
    assert sims[0] > sims[1]
