"""Unit tests for dflash.model helpers and the dflash_generate stop-token check.

These tests intentionally cover only the pure-Python / pure-tensor logic so they
run on CPU without needing weights or transformers Qwen3 wiring.
"""

import pytest

torch = pytest.importorskip("torch")

from dflash.model import build_target_layer_ids, extract_context_feature, sample


def test_build_target_layer_ids_single_draft_layer():
    assert build_target_layer_ids(num_target_layers=24, num_draft_layers=1) == [12]


def test_build_target_layer_ids_endpoints_for_two_draft_layers():
    layers = build_target_layer_ids(num_target_layers=64, num_draft_layers=2)
    assert layers == [1, 64 - 3]


def test_build_target_layer_ids_evenly_interpolates():
    layers = build_target_layer_ids(num_target_layers=64, num_draft_layers=4)
    assert layers[0] == 1
    assert layers[-1] == 64 - 3
    assert len(layers) == 4
    assert layers == sorted(layers)


def test_extract_context_feature_concatenates_offset_layers():
    # extract_context_feature reads hidden_states[layer_id + 1] for each layer_id
    # (the +1 offset skips the embedding output).
    bsz, seq, hidden = 2, 5, 8
    hidden_states = [torch.full((bsz, seq, hidden), float(i)) for i in range(6)]
    out = extract_context_feature(hidden_states, layer_ids=[0, 2, 4])
    assert out.shape == (bsz, seq, hidden * 3)
    # hidden_states[1] is all 1.0, hidden_states[3] is all 3.0, hidden_states[5] is all 5.0
    assert torch.equal(out[..., :hidden], torch.full((bsz, seq, hidden), 1.0))
    assert torch.equal(out[..., hidden : 2 * hidden], torch.full((bsz, seq, hidden), 3.0))
    assert torch.equal(out[..., 2 * hidden :], torch.full((bsz, seq, hidden), 5.0))


def test_sample_temperature_zero_is_argmax():
    logits = torch.tensor([[[2.0, 1.5, 1.0, 0.5], [0.1, 0.9, 0.2, 0.0]]])
    out = sample(logits, temperature=0.0)
    assert out.shape == (1, 2)
    assert out[0, 0].item() == 0
    assert out[0, 1].item() == 1


def test_sample_with_temperature_returns_in_range():
    torch.manual_seed(0)
    logits = torch.randn(1, 4, 32)
    out = sample(logits, temperature=1.0)
    assert out.shape == (1, 4)
    assert (out >= 0).all() and (out < 32).all()


def _legacy_in_loop_check(output_ids, num_input_tokens, stop_token_ids):
    """Reproduces the pre-fix in-loop early-exit check from dflash_generate."""
    return any(
        stop_token_id in output_ids[:, num_input_tokens:]
        for stop_token_id in stop_token_ids
    )


def _new_in_loop_check(output_ids, num_input_tokens, cursor, stop_token_tensor):
    """The new in-loop early-exit check, scoped to the actually written slice."""
    return torch.isin(
        output_ids[0, num_input_tokens : cursor + 1], stop_token_tensor
    ).any().item()


def test_stop_token_check_does_not_scan_uninitialized_buffer():
    """Regression test for the pre-fix in-loop check.

    dflash_generate pre-allocates output_ids with mask_token_id past the
    cursor. If mask_token_id is also a stop token (a model-config-dependent
    edge case the maintainer already cares about — see PR #76 "Preserve
    output tokens that equal mask_token_id"), the legacy
    `stop_id in output_ids[:, num_input_tokens:]` check fires on every
    iteration because the still-pristine tail of the buffer is full of
    mask tokens equal to a stop token. Generation aborts after the first
    block even though no real stop was emitted.

    The new check restricts the scan to positions that have actually been
    written (`[num_input_tokens, cursor + 1]`) and does not regress.
    """
    mask_token_id = 99
    stop_token_ids = [99, 1]  # collides with mask_token_id
    num_input_tokens = 10
    max_length = 50
    block_size = 8

    output_ids = torch.full(
        (1, max_length + block_size), mask_token_id, dtype=torch.long
    )
    output_ids[:, :num_input_tokens] = 5

    # Simulate one block of generation: 4 accepted draft tokens + 1 bonus,
    # none of which are stop tokens. After dflash_generate's
    # `start += acceptance_length + 1`, the cursor lands on the bonus token.
    written = torch.tensor([5, 6, 7, 8, 10])  # no stop tokens, no mask tokens
    output_ids[0, num_input_tokens : num_input_tokens + written.numel()] = written
    cursor = num_input_tokens + written.numel() - 1

    # Pre-fix check spuriously fires because it scans the still-mask-filled
    # tail past the cursor.
    assert _legacy_in_loop_check(output_ids, num_input_tokens, stop_token_ids) is True

    stop_tensor = torch.as_tensor(stop_token_ids, dtype=output_ids.dtype)
    assert _new_in_loop_check(output_ids, num_input_tokens, cursor, stop_tensor) is False


def test_stop_token_check_detects_real_stop_after_cursor_advance():
    """When a real stop token is written and the cursor has advanced, the
    new check fires (matching the legacy semantic for the common case)."""
    mask_token_id = 0
    stop_token_ids = [7, 11]
    num_input_tokens = 4
    max_length = 32
    block_size = 4

    output_ids = torch.full(
        (1, max_length + block_size), mask_token_id, dtype=torch.long
    )
    output_ids[:, :num_input_tokens] = 5

    # Simulate one block worth of generation; cursor advances by 4.
    output_ids[0, num_input_tokens : num_input_tokens + 4] = torch.tensor([3, 7, 6, 8])
    cursor = num_input_tokens + 3

    stop_tensor = torch.as_tensor(stop_token_ids, dtype=output_ids.dtype)
    assert _new_in_loop_check(output_ids, num_input_tokens, cursor, stop_tensor) is True
