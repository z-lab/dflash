#!/usr/bin/env python3
import json
import traceback
from huggingface_hub import hf_hub_download
from dflash.model_mlx import load, load_draft, stream_generate


MODEL_ID = "Qwen/Qwen3-Coder-Next"
DRAFT_ID = "z-lab/Qwen3-Coder-Next-DFlash"
PROMPT = """You are a coding assistant.
Write a Python function `is_prime(n: int) -> bool` and include 3 small tests."""


def get_total_size_gb(model_id: str) -> float | None:
    try:
        idx = hf_hub_download(model_id, "model.safetensors.index.json")
        with open(idx, "r", encoding="utf-8") as f:
            data = json.load(f)
        total_size = data.get("metadata", {}).get("total_size")
        if total_size is None:
            return None
        return total_size / (1024**3)
    except Exception:
        return None


def main() -> None:
    print(f"[INFO] target model: {MODEL_ID}")
    print(f"[INFO] draft model : {DRAFT_ID}")
    total_size_gb = get_total_size_gb(MODEL_ID)
    if total_size_gb is not None:
        print(f"[INFO] target safetensors total size: {total_size_gb:.2f} GiB")

    try:
        print("[INFO] Loading target model with MLX...")
        model, tokenizer = load(MODEL_ID)
        print("[INFO] Loading DFlash draft model...")
        draft = load_draft(DRAFT_ID)
        print("[INFO] Starting generation...")

        out_text = []
        tps = 0.0
        for r in stream_generate(
            model=model,
            draft=draft,
            tokenizer=tokenizer,
            prompt=PROMPT,
            block_size=16,
            max_tokens=128,
            temperature=0.2,
        ):
            if r.text:
                print(r.text, end="", flush=True)
                out_text.append(r.text)
            tps = r.generation_tps
            if r.finish_reason is not None:
                print(f"\n[INFO] finish_reason={r.finish_reason}")
                break

        print(f"[INFO] generation_tps={tps:.2f} tok/s")
    except Exception as e:
        print("\n[ERROR] Generation failed.")
        print(f"[ERROR] {type(e).__name__}: {e}")
        print("[ERROR] This often indicates local memory/VRAM is insufficient for this model pair.")
        traceback.print_exc()


if __name__ == "__main__":
    main()
