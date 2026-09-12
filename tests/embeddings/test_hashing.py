"""HashingEmbedder: deterministic, download-free, protocol-conformant."""

from __future__ import annotations

import numpy as np
import pytest

from secqa.core.contracts import Embedder
from secqa.embeddings import HashingEmbedder
from secqa.embeddings.base import l2_normalise

SENTENCE = "Total net sales were $1,577 million in fiscal 2023, an increase of 12% over 2022."
PARAPHRASE = "Net sales in fiscal 2023 totalled $1,577 million, up 12% versus 2022."
UNRELATED = "A quick brown fox jumps across a lazy dog near a quiet riverbank."


def test_defaults_and_protocol() -> None:
    embedder = HashingEmbedder()
    assert embedder.name == "hashing-384"
    assert embedder.dim == 384
    assert isinstance(embedder, Embedder)


def test_shape_dtype_and_unit_norm() -> None:
    vectors = HashingEmbedder().embed([SENTENCE, PARAPHRASE, UNRELATED])
    assert vectors.shape == (3, 384)
    assert vectors.dtype == np.float32
    assert vectors.flags["C_CONTIGUOUS"]
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-6)


def test_deterministic_across_instances_and_batch_sizes() -> None:
    texts = [SENTENCE, PARAPHRASE, UNRELATED, "cash and cash equivalents"]
    a = HashingEmbedder().embed(texts)
    b = HashingEmbedder().embed(texts, batch_size=1)
    c = HashingEmbedder(dim=384, seed=0).embed(texts, batch_size=3)
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(a, c)


def test_paraphrase_with_shared_words_closer_than_unrelated() -> None:
    v = HashingEmbedder().embed([SENTENCE, PARAPHRASE, UNRELATED])
    sim_paraphrase = float(v[0] @ v[1])
    sim_unrelated = float(v[0] @ v[2])
    assert sim_paraphrase > 0.2
    assert sim_paraphrase > sim_unrelated
    assert sim_unrelated < 0.1


def test_query_and_passage_kinds_are_identical() -> None:
    embedder = HashingEmbedder()
    np.testing.assert_array_equal(
        embedder.embed([SENTENCE], kind="query"), embedder.embed([SENTENCE], kind="passage")
    )


def test_seed_changes_the_space_but_stays_deterministic() -> None:
    base = HashingEmbedder().embed([SENTENCE])
    seeded = HashingEmbedder(seed=7).embed([SENTENCE])
    seeded_again = HashingEmbedder(seed=7).embed([SENTENCE])
    assert not np.allclose(base, seeded)
    np.testing.assert_array_equal(seeded, seeded_again)


def test_custom_dim() -> None:
    embedder = HashingEmbedder(dim=64)
    assert embedder.name == "hashing-64"
    assert embedder.embed(["net income"]).shape == (1, 64)


def test_blank_texts_embed_to_zero_vectors_in_place() -> None:
    vectors = HashingEmbedder().embed(["", SENTENCE, "   \n"])
    assert not vectors[0].any()
    assert not vectors[2].any()
    assert np.isclose(np.linalg.norm(vectors[1]), 1.0)
    assert vectors.shape == (3, 384)


def test_empty_input_gives_empty_matrix() -> None:
    vectors = HashingEmbedder().embed([])
    assert vectors.shape == (0, 384)
    assert vectors.dtype == np.float32


def test_case_and_accent_insensitive() -> None:
    embedder = HashingEmbedder()
    v = embedder.embed(["Net Income", "net income", "nét income"])
    np.testing.assert_allclose(v[0], v[1], atol=1e-7)
    np.testing.assert_allclose(v[0], v[2], atol=1e-7)


@pytest.mark.parametrize("dim", [0, -5, 3.5, True])
def test_invalid_dim_rejected(dim: object) -> None:
    with pytest.raises(ValueError):
        HashingEmbedder(dim=dim)  # type: ignore[arg-type]


def test_invalid_inputs_rejected() -> None:
    embedder = HashingEmbedder()
    with pytest.raises(TypeError):
        embedder.embed("a single string")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        embedder.embed(["ok", 42])  # type: ignore[list-item]
    with pytest.raises(ValueError):
        embedder.embed(["ok"], batch_size=0)
    with pytest.raises(ValueError):
        embedder.embed(["ok"], kind="document")  # type: ignore[arg-type]


def test_l2_normalise_helper() -> None:
    out = l2_normalise(np.array([[3.0, 4.0], [0.0, 0.0]]))
    np.testing.assert_allclose(out, [[0.6, 0.8], [0.0, 0.0]])
    assert out.dtype == np.float32
    with pytest.raises(ValueError):
        l2_normalise(np.array([1.0, 2.0]))
