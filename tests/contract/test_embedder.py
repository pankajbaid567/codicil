from pathlib import Path

import numpy as np
import pytest

from codicil.adapters.hashing_embedder import HashingEmbedder

MODELS = Path(__file__).resolve().parents[2] / "data" / "models"


def embedders():
    out = [pytest.param(HashingEmbedder(), id="hashing")]
    for name in ("multilingual-e5-small", "bge-small-en-v1.5"):
        if (MODELS / name / "onnx" / "model_quantized.onnx").exists():
            from codicil.adapters.onnx_embedder import OnnxEmbedder

            out.append(pytest.param(OnnxEmbedder(MODELS / name), id=name))
    return out


@pytest.mark.parametrize("emb", embedders())
def test_unit_norm_dim_and_determinism(emb):
    v = emb.embed_passages(["The bank shall credit the amount.", "Interest on savings deposits."])
    assert v.shape == (2, emb.dim) and v.dtype == np.float32
    assert np.allclose(np.linalg.norm(v, axis=1), 1.0, atol=1e-3)
    assert np.allclose(v, emb.embed_passages(["The bank shall credit the amount.", "Interest on savings deposits."]), atol=1e-5)


@pytest.mark.parametrize("emb", embedders())
def test_empty_input_and_related_text_is_closer(emb):
    assert emb.embed_queries([]).shape == (0, emb.dim)
    q = emb.embed_queries(["bank reverses unauthorised electronic transaction"])[0]
    near, far = emb.embed_passages(
        ["The bank shall reverse an unauthorised electronic transaction within ten days.", "Peer to peer lending platform leverage ratio."]
    )
    assert q @ near > q @ far


@pytest.mark.skipif(not (MODELS / "multilingual-e5-small" / "onnx" / "model_quantized.onnx").exists(), reason="model not downloaded")
def test_e5_queries_and_passages_use_different_prefixes():
    from codicil.adapters.onnx_embedder import OnnxEmbedder

    emb = OnnxEmbedder(MODELS / "multilingual-e5-small")
    text = "unauthorised electronic transaction"
    assert not np.allclose(emb.embed_queries([text]), emb.embed_passages([text]), atol=1e-4)
