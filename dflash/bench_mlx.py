"""CLI benchmark for DFlash speculative decoding against plain mlx_lm generation."""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path

import mlx.core as mx

from .benchmark import apply_chat_template, load_mlx_models
from .model_mlx import load as load_target
from .model_mlx import make_sampler
from .model_mlx import stream_generate as dflash_stream_generate

DEFAULT_PROMPTS = Path(__file__).parent / "bench_prompts.json"


def _parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark DFlash speculative decoding on MLX.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--draft", help="Draft model id. Omit to run plain mlx_lm greedy/sampled generation.")
    parser.add_argument("--draft-bits", type=int, choices=[4, 8])
    parser.add_argument("--block-sizes", help="Comma-separated block sizes. Ignored without --draft.")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--prompts", help="JSON file with a list of chat prompt strings.")
    parser.add_argument("--reasoning", help="Passed through to apply_chat_template.")
    parser.add_argument("--out", required=True, help="JSONL output path.")
    parser.add_argument(
        "--baseline",
        help="A prior baseline JSONL to check greedy token shas against, "
        "for runs where this run has no draft-less rows of its own.",
    )
    return parser.parse_args(argv)


def _load_prompts(path: str | None) -> list[str]:
    return json.loads(Path(path or DEFAULT_PROMPTS).read_text())


def _sha256_tokens(tokens: list[int]) -> str:
    return hashlib.sha256(json.dumps(tokens).encode()).hexdigest()


def _load_baseline_shas(path: str | None) -> dict[int, str]:
    if not path:
        return {}
    shas = {}
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("draft") is None:
            shas[row["prompt_index"]] = row["token_sha256"]
    return shas


def _run_baseline(model, tokenizer, prompt_text: str, args: argparse.Namespace) -> dict:
    from mlx_lm import stream_generate as mlx_stream_generate

    sampler = make_sampler(args.temperature, args.top_p, args.top_k)
    tokens, text_parts, last = [], [], None
    for r in mlx_stream_generate(model, tokenizer, prompt_text, max_tokens=args.max_tokens, sampler=sampler):
        tokens.append(r.token)
        text_parts.append(r.text)
        last = r
    return {
        "prompt_tokens": last.prompt_tokens,
        "gen_tokens": len(tokens),
        "gen_tps": last.generation_tps,
        "prompt_tps": last.prompt_tps,
        "mean_accept": None,
        "peak_gb": last.peak_memory,
        "tokens": tokens,
        "text": "".join(text_parts),
    }


def _run_draft(model, draft, tokenizer, prompt_text: str, block_size: int, args: argparse.Namespace) -> dict:
    tokens, text_parts, accepted, last = [], [], [], None
    for r in dflash_stream_generate(
        model, draft, tokenizer, prompt_text,
        block_size=block_size,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
    ):
        tokens.extend(r.tokens)
        text_parts.append(r.text)
        if r.accepted is not None:
            accepted.append(r.accepted)
        last = r
    return {
        "prompt_tokens": last.prompt_tokens,
        "gen_tokens": len(tokens),
        "gen_tps": last.generation_tps,
        "prompt_tps": last.prompt_tps,
        "mean_accept": statistics.mean(accepted) if accepted else None,
        "peak_gb": last.peak_memory,
        "tokens": tokens,
        "text": "".join(text_parts),
    }


def _print_summary(rows: list[dict], baseline_shas: dict[int, str], temperature: float) -> None:
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        groups.setdefault((row["draft"], row["block_size"]), []).append(row)

    print(f"\n{'draft':<50} {'block':>6} {'mean_tps':>10} {'mean_accept':>12} {'greedy_match':>14}")
    for (draft, block_size), group in sorted(groups.items(), key=lambda kv: (kv[0][0] or "", kv[0][1] or 0)):
        mean_tps = statistics.mean(r["gen_tps"] for r in group)
        accepts = [r["mean_accept"] for r in group if r["mean_accept"] is not None]
        mean_accept = f"{statistics.mean(accepts):.2f}" if accepts else "n/a"
        match = "n/a"
        if temperature == 0 and baseline_shas:
            mismatches = [
                r["prompt_index"] for r in group
                if r["prompt_index"] in baseline_shas and baseline_shas[r["prompt_index"]] != r["token_sha256"]
            ]
            match = "yes" if not mismatches else f"no {mismatches}"
        print(f"{str(draft):<50} {str(block_size):>6} {mean_tps:>10.2f} {mean_accept:>12} {match:>14}")


def main(argv=None) -> None:
    args = _parse_args(argv)
    mx.random.seed(0)
    prompts = _load_prompts(args.prompts)

    if args.draft:
        model, draft, tokenizer = load_mlx_models(args.model, args.draft, args.draft_bits)
        block_sizes = (
            [int(x) for x in args.block_sizes.split(",")]
            if args.block_sizes
            else [int(draft.config.block_size)]
        )
    else:
        model, tokenizer = load_target(args.model)
        draft = None
        block_sizes = [None]

    known_shas = _load_baseline_shas(args.baseline)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    with out_path.open("w") as fh:
        for prompt_index, prompt_str in enumerate(prompts):
            messages = [{"role": "user", "content": prompt_str}]
            prompt_text = apply_chat_template(tokenizer, messages, args.reasoning)
            for block_size in block_sizes:
                tic = time.perf_counter()
                result = (
                    _run_baseline(model, tokenizer, prompt_text, args)
                    if draft is None
                    else _run_draft(model, draft, tokenizer, prompt_text, block_size, args)
                )
                token_sha = _sha256_tokens(result["tokens"])
                if draft is None:
                    known_shas.setdefault(prompt_index, token_sha)
                row = {
                    "model": args.model,
                    "draft": args.draft,
                    "draft_bits": args.draft_bits,
                    "block_size": block_size,
                    "temperature": args.temperature,
                    "prompt_index": prompt_index,
                    "prompt_tokens": result["prompt_tokens"],
                    "gen_tokens": result["gen_tokens"],
                    "gen_tps": result["gen_tps"],
                    "prompt_tps": result["prompt_tps"],
                    "mean_accept": result["mean_accept"],
                    "peak_gb": result["peak_gb"],
                    "token_sha256": token_sha,
                    "tokens": result["tokens"],
                    "text_preview": result["text"][:120],
                    "wall_s": time.perf_counter() - tic,
                }
                fh.write(json.dumps(row) + "\n")
                fh.flush()
                rows.append(row)

    _print_summary(rows, known_shas, args.temperature)


if __name__ == "__main__":
    main()
