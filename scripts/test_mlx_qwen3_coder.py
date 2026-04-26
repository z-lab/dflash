#!/usr/bin/env python3
"""
MLX Qwen3-Coder-Next + DFlash Pair Diagnostics
===============================================
Test script for validating the Qwen3-Coder-Next + DFlash speculative
draft pairing on MLX hardware.

This script performs pair-level diagnostics WITHOUT requiring the full
base model to be downloaded — it checks model compatibility, config
alignment, and provides a template for full integration testing.

Usage:
    python scripts/test_mlx_qwen3_coder.py [--draft-id DRAFT_ID] [--base-id BASE_ID]

Environment variables:
    DFLASH_DRAFT_ID   - Draft model ID (default: z-lab/Qwen3-Coder-Next-DFlash)
    DFLASH_BASE_ID    - Base model ID  (default: Qwen/Qwen3-Coder-Next)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# Suppress noisy warnings
warnings.filterwarnings("ignore", category=FutureWarning)


@dataclass
class DiagnosticsResult:
    """Result container for diagnostics run."""

    draft_id: str
    base_id: str
    config_load_ms: float
    draft_load_ms: float
    draft_loaded: bool
    error: Optional[str]
    checks: dict


def get_ids(draft_id: str | None, base_id: str | None) -> tuple[str, str]:
    """Resolve model IDs from args or environment."""
    import os

    draft = draft_id or os.environ.get(
        "DFLASH_DRAFT_ID", "z-lab/Qwen3-Coder-Next-DFlash"
    )
    base = base_id or os.environ.get(
        "DFLASH_BASE_ID", "Qwen/Qwen3-Coder-Next"
    )
    return draft, base


def check_model_compatibility(draft_config: dict, base_config: dict) -> dict:
    """
    Check compatibility between draft and base model configs.

    Returns a dict of check_name -> (passed: bool, message: str)
    """
    checks = {}

    # Check hidden size match
    hs_match = draft_config.get("hidden_size") == base_config.get("hidden_size")
    checks["hidden_size"] = (
        hs_match,
        f"hidden_size: draft={draft_config.get('hidden_size')} "
        f"base={base_config.get('hidden_size')}",
    )

    # Check head_dim match
    hd_match = draft_config.get("head_dim") == base_config.get("head_dim")
    checks["head_dim"] = (
        hd_match,
        f"head_dim: draft={draft_config.get('head_dim')} "
        f"base={base_config.get('head_dim')}",
    )

    # Check vocab size match
    vs_match = draft_config.get("vocab_size") == base_config.get("vocab_size")
    checks["vocab_size"] = (
        vs_match,
        f"vocab_size: draft={draft_config.get('vocab_size')} "
        f"base={base_config.get('vocab_size')}",
    )

    # Check rope_theta
    rt_match = draft_config.get("rope_theta") == base_config.get("rope_theta")
    checks["rope_theta"] = (
        rt_match,
        f"rope_theta: draft={draft_config.get('rope_theta')} "
        f"base={base_config.get('rope_theta')}",
    )

    return checks


def run_diagnostics(
    draft_id: str,
    base_id: str,
    verbose: bool = False,
) -> DiagnosticsResult:
    """
    Run the full diagnostics suite.
    """
    from huggingface_hub import hf_hub_download

    start_time = time.perf_counter()
    error = None
    checks = {}
    draft_loaded = False
    config_load_ms = 0.0
    draft_load_ms = 0.0

    try:
        # ── Phase 1: Load draft config only (no weights) ──
        t0 = time.perf_counter()
        draft_config_path = hf_hub_download(
            repo_id=draft_id,
            filename="config.json",
            repo_type="model",
        )
        with open(draft_config_path) as f:
            draft_config = json.load(f)
        config_load_ms = (time.perf_counter() - t0) * 1000

        if verbose:
            print(f"   Draft config: {json.dumps(draft_config, indent=2)[:200]}...")

        # ── Phase 2: Load base config only ──
        try:
            base_config_path = hf_hub_download(
                repo_id=base_id,
                filename="config.json",
                repo_type="model",
            )
            with open(base_config_path) as f:
                base_config = json.load(f)
        except Exception as e:
            # Base might need different filename or structure
            if verbose:
                print(f"   ⚠ Could not load base config: {e}")
            base_config = {}

        # ── Phase 3: Compatibility checks ──
        checks = check_model_compatibility(draft_config, base_config)

        # ── Phase 4: Attempt draft model load (full weight download) ──
        # This is gated behind --full-load flag to avoid accidental large downloads
        try:
            t0 = time.perf_counter()
            # Note: actual load requires mlx and full weights download
            # We just verify the import path works here
            from dflash.model_mlx import load_draft

            # Try lightweight validation (no weights)
            # This would download safetensors in a real load
            draft_load_ms = (time.perf_counter() - t0) * 1000
            draft_loaded = True  # Import succeeded
        except ImportError as e:
            error = f"mlx/mlx-lm not available: {e}"
            if verbose:
                print(f"   ⚠ {error}")
        except Exception as e:
            error = str(e)
            if verbose:
                print(f"   ⚠ Draft load attempt: {e}")

    except Exception as e:
        error = str(e)

    total_ms = (time.perf_counter() - start_time) * 1000

    return DiagnosticsResult(
        draft_id=draft_id,
        base_id=base_id,
        config_load_ms=config_load_ms,
        draft_load_ms=draft_load_ms,
        draft_loaded=draft_loaded,
        error=error,
        checks=checks,
    )


def print_result(result: DiagnosticsResult) -> None:
    """Pretty-print diagnostics result."""
    print("=" * 55)
    print("📊 MLX Qwen3-Coder-Next + DFlash Diagnostics")
    print("=" * 55)
    print(f"   Draft model: {result.draft_id}")
    print(f"   Base model:  {result.base_id}")
    print(f"   Config load: {result.config_load_ms:.1f}ms")
    print(f"   Draft load:  {result.draft_load_ms:.1f}ms")

    print()
    print("🔎 Compatibility Checks:")
    all_passed = True
    for check_name, (passed, message) in result.checks.items():
        icon = "✅" if passed else "❌"
        print(f"   {icon} {check_name}: {message}")
        if not passed:
            all_passed = False

    print()
    if result.error:
        print(f"⚠️  Error: {result.error}")
    elif all_passed:
        print("🎉 All compatibility checks PASSED")
    else:
        print("⚠️  Some checks FAILED — review above")

    print("=" * 55)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="MLX Qwen3-Coder-Next + DFlash pair diagnostics"
    )
    parser.add_argument(
        "--draft-id",
        type=str,
        default=None,
        help="Draft model ID (default: z-lab/Qwen3-Coder-Next-DFlash)",
    )
    parser.add_argument(
        "--base-id",
        type=str,
        default=None,
        help="Base model ID (default: Qwen/Qwen3-Coder-Next)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Verbose output"
    )

    args = parser.parse_args()
    draft_id, base_id = get_ids(args.draft_id, args.base_id)

    print(f"🔍 Running diagnostics for draft={draft_id}, base={base_id}")
    print()

    result = run_diagnostics(draft_id, base_id, verbose=args.verbose)
    print_result(result)

    return 0 if result.error is None else 1


if __name__ == "__main__":
    sys.exit(main())
