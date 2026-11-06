import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from dflash.model import DFlash2DraftModel, _grouped_dynamic_convolve, dflash_generate


@pytest.mark.parametrize("length", [1, 2, 3, 4, 16])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_convolution_matches_prefix_of_longer_block(length, dtype):
    torch.manual_seed(19)
    hidden = torch.randn(2, 16, 32, dtype=dtype)
    dynamic = torch.randn(2, 16, 4, 2, dtype=dtype)
    base = torch.randn(4, 32, dtype=dtype)
    expected = _grouped_dynamic_convolve(hidden, dynamic, base, 16)[:, :length]
    actual = _grouped_dynamic_convolve(hidden[:, :length], dynamic[:, :length], base, 16)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("length", [1, 2, 3])
def test_short_convolution_gradients_match_longer_block(length):
    torch.manual_seed(19)
    hidden = torch.randn(2, 8, 32, dtype=torch.float64, requires_grad=True)
    dynamic = torch.randn(2, 8, 4, 2, dtype=torch.float64, requires_grad=True)
    base = torch.randn(4, 32, dtype=torch.float64, requires_grad=True)
    full = _grouped_dynamic_convolve(hidden, dynamic, base, 16)[:, :length]
    short = _grouped_dynamic_convolve(hidden[:, :length], dynamic[:, :length], base, 16)
    expected = torch.autograd.grad(full.square().sum(), (hidden, dynamic, base))
    actual = torch.autograd.grad(short.square().sum(), (hidden, dynamic, base))
    for actual_gradient, expected_gradient in zip(actual, expected):
        torch.testing.assert_close(actual_gradient, expected_gradient)


@pytest.mark.parametrize("max_new_tokens", [1, 2, 3, 4, 17])
@torch.inference_mode()
def test_dflash2_short_output_budget_matches_autoregressive(max_new_tokens):
    torch.manual_seed(19)
    config = Qwen3Config(
        vocab_size=97,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        pad_token_id=0,
        eos_token_id=None,
    )
    target = Qwen3ForCausalLM(config).eval()
    draft_config = Qwen3Config(**config.to_dict())
    draft_config.num_target_layers = 2
    draft_config.dflash_config = {
        "block_size": 16,
        "mask_token_id": 96,
        "target_layer_ids": [0, 1],
        "conv_kernel_size": 4,
        "conv_group_size": 16,
        "selector_rank": 8,
        "selector_top_k": 8,
    }
    draft = DFlash2DraftModel(draft_config).eval()
    # Fresh convolution modules have uninitialized base kernels until weights load.
    for layer in draft.layers:
        layer.attention_conv.base_kernel.normal_(std=0.02)
        layer.mlp_conv.base_kernel.normal_(std=0.02)
    prompt = torch.tensor([[3, 4, 5, 6]])
    expected = target.generate(prompt, max_new_tokens=max_new_tokens, do_sample=False)
    actual = dflash_generate(draft, target, prompt, max_new_tokens, stop_token_ids=None)
    assert actual.shape[1] == prompt.shape[1] + max_new_tokens
    torch.testing.assert_close(actual, expected)
