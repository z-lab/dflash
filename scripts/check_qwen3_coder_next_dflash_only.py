#!/usr/bin/env python3
from dflash.model_mlx import load_draft


def main() -> None:
    draft_id = "z-lab/Qwen3-Coder-Next-DFlash"
    draft = load_draft(draft_id)
    cfg = draft.config
    print("[OK] DFlash draft loaded")
    print(f"draft_id={draft_id}")
    print(f"block_size={cfg.block_size}")
    print(f"num_target_layers={cfg.num_target_layers}")
    print(f"target_layer_ids={list(cfg.target_layer_ids)}")
    print("[NOTE] This script does NOT download Qwen/Qwen3-Coder-Next.")


if __name__ == "__main__":
    main()
