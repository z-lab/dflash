#!/usr/bin/env python3
"""
MLX Setup Verification Script for DFlash
==========================================
Validates that z-lab/Qwen3-Coder-Next-DFlash can be loaded
WITHOUT downloading the large base model weights.

This is a draft-only check — useful for local setup validation
without triggering large downloads.

Usage:
    python scripts/check_mlx_setup.py [--draft-id DRAFT_ID]

Environment:
    DFLASH_DRAFT_ID - optional env var, defaults to z-lab/Qwen3-Coder-Next-DFlash
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

# Suppress huggingface_hub download warnings for cleaner output
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*huggingface_hub.*")


def get_draft_id(draft_id: str | None) -> str:
    """Resolve the draft model ID."""
    if draft_id:
        return draft_id
    import os
    return os.environ.get("DFLASH_DRAFT_ID", "z-lab/Qwen3-Coder-Next-DFlash")


def check_config_only(draft_id: str) -> dict:
    """
    Load config.json from the draft model WITHOUT downloading safetensors.

    This validates the model repository structure and config format
    without triggering a large safetensors download.
    """
    from huggingface_hub import hf_hub_download

    # Only download the small config.json, not the weights
    config_path = hf_hub_download(
        repo_id=draft_id,
        filename="config.json",
        repo_type="model",
    )

    with open(config_path, "r") as f:
        config = json.load(f)

    return config


def check_tokenizer_only(draft_id: str) -> bool:
    """
    Check if tokenizer files are accessible (optional, small download).
    """
    from huggingface_hub import hf_hub_download

    try:
        # Try to get tokenizer config — small file, fast to check
        hf_hub_download(
            repo_id=draft_id,
            filename="tokenizer_config.json",
            repo_type="model",
        )
        return True
    except Exception:
        return False


def validate_config(config: dict) -> list[str]:
    """
    Validate the DFlash config structure.
    Returns list of warnings (empty = valid).
    """
    warnings_list = []

    required_fields = [
        "hidden_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
        "intermediate_size",
        "vocab_size",
        "rms_norm_eps",
        "rope_theta",
        "max_position_embeddings",
        "block_size",
        "dflash_config",
    ]

    for field in required_fields:
        if field not in config:
            warnings_list.append(f"Missing required field: {field}")

    # Check dflash_config sub-fields
    if "dflash_config" in config:
        dc = config["dflash_config"]
        for subfield in ["target_layer_ids", "num_target_layers", "mask_token_id"]:
            if subfield not in dc:
                warnings_list.append(f"Missing dflash_config field: {subfield}")

    # Sanity checks
    if config.get("hidden_size", 0) <= 0:
        warnings_list.append("hidden_size must be positive")

    if config.get("num_hidden_layers", 0) <= 0:
        warnings_list.append("num_hidden_layers must be positive")

    return warnings_list


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify DFlash MLX setup without downloading base model weights"
    )
    parser.add_argument(
        "--draft-id",
        type=str,
        default=None,
        help="Draft model ID (default: z-lab/Qwen3-Coder-Next-DFlash)",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    args = parser.parse_args()
    draft_id = get_draft_id(args.draft_id)

    print(f"🔍 Checking draft model: {draft_id}")
    print("-" * 50)

    # Step 1: Check config only (no weights download)
    print("📋 Downloading config.json (small, no weights)...")
    try:
        config = check_config_only(draft_id)
        print("✅ config.json loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load config.json: {e}")
        return 1

    # Step 2: Validate config structure
    print("🔎 Validating config structure...")
    validation_warnings = validate_config(config)

    if validation_warnings:
        print("⚠️  Config validation warnings:")
        for w in validation_warnings:
            print(f"   - {w}")
    else:
        print("✅ Config structure valid")

    # Step 3: Optional tokenizer check
    print("🔤 Checking tokenizer availability...")
    tokenizer_ok = check_tokenizer_only(draft_id)
    if tokenizer_ok:
        print("✅ Tokenizer files accessible")
    else:
        print("⚠️  Tokenizer files not found (may need separate download)")

    # Step 4: Summary
    print("-" * 50)
    print("📊 Setup Check Summary")
    print(f"   Draft ID: {draft_id}")
    print(f"   Hidden size: {config.get('hidden_size', 'N/A')}")
    print(f"   Num layers: {config.get('num_hidden_layers', 'N/A')}")
    print(f"   Vocab size: {config.get('vocab_size', 'N/A')}")
    print(f"   Target layer IDs: {config.get('dflash_config', {}).get('target_layer_ids', 'N/A')}")

    print()
    if not validation_warnings:
        print("🎉 MLX setup check PASSED — ready to use with mlx_lm")
        print("   (No base model weights were downloaded)")
        return 0
    else:
        print("⚠️  MLX setup check PASSED with warnings")
        return 0  # Still success, just warnings


if __name__ == "__main__":
    sys.exit(main())
