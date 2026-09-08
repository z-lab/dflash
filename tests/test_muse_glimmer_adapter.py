"""Draft-adapter unit tests. No weights, no network: config/remap logic only.

Fixtures under tests/fixtures/ are verbatim copies of the real checkpoints'
config.json and safetensors headers (see reference/README.md in the port
workspace for provenance).
"""
import json
from pathlib import Path

import pytest

mx = pytest.importorskip("mlx.core")
from mlx.utils import tree_flatten

from dflash.model_mlx import (
    DFlash2DraftModel,
    DFlashDraftModel,
    _draft_config,
    _draft_model_class,
    _remap_draft_weights,
)

FIXTURES = Path(__file__).parent / "fixtures"


def _load(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text())


def _tensor_names(header: dict) -> list[str]:
    return [k for k in header if k != "__metadata__"]


def _param_keys(model) -> set[str]:
    return {k for k, _ in tree_flatten(model.parameters())}


def test_meta_v1_config():
    cfg = _load("assistant-config.json")
    config = _draft_config(cfg)

    assert config.block_size == 16
    assert config.mask_token_id == 201818
    assert config.target_layer_ids == (1, 13, 25, 37, 49)
    assert config.vocab_size == 202048
    assert config.is_causal is False
    assert config.sliding_window == 2048
    assert config.layer_types == ("sliding_attention",) * 5
    assert config.num_hidden_layers == 5
    assert config.hidden_size == 6656
    assert config.rope_theta == 500000.0


def test_meta_v1_architecture_selects_dflash_draft_model():
    cfg = _load("assistant-config.json")
    assert _draft_model_class(cfg) is DFlashDraftModel


def test_meta_v1_weight_remap_covers_every_model_parameter():
    cfg = _load("assistant-config.json")
    header = _load("assistant-safetensors-header.json")
    synthetic = {name: mx.zeros((1,)) for name in _tensor_names(header)}

    remapped = _remap_draft_weights(cfg, synthetic)
    model = DFlashDraftModel(_draft_config(cfg))

    # embed_tokens/lm_head are bound from the target later and hold no
    # parameters of their own until then, so they never appear here.
    assert set(remapped.keys()) == _param_keys(model)


def test_dflash2_config():
    cfg = _load("dflash2-config.json")
    config = _draft_config(cfg)

    assert config.block_size == 16
    assert config.mask_token_id == 201818
    assert config.target_layer_ids == (1, 13, 25, 37, 49)
    assert config.vocab_size == 202048
    assert config.is_causal is False
    assert config.selector_rank == 256
    assert config.selector_top_k == 16
    assert config.conv_kernel_size == 2
    assert config.conv_group_size == 16
    assert config.num_target_layers == 52


def test_dflash2_architecture_selects_dflash2_draft_model():
    cfg = _load("dflash2-config.json")
    assert _draft_model_class(cfg) is DFlash2DraftModel


def test_dflash2_weight_remap_covers_every_model_parameter():
    cfg = _load("dflash2-config.json")
    header = _load("dflash2-safetensors-header.json")
    synthetic = {name: mx.zeros((1,)) for name in _tensor_names(header)}

    remapped = _remap_draft_weights(cfg, synthetic)
    model = DFlash2DraftModel(_draft_config(cfg))

    assert set(remapped.keys()) == _param_keys(model)


def test_unknown_architecture_raises():
    with pytest.raises(ValueError, match="SomeOtherModel"):
        _draft_model_class({"architectures": ["SomeOtherModel"]})
