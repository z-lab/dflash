from __future__ import annotations

import argparse
import random
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import chain
from types import SimpleNamespace

import requests
from tqdm import tqdm

DATASETS = {
    "gsm8k": {
        "load_args": ("openai/gsm8k", "main"),
        "load_kwargs": {"split": "test"},
        "format": lambda x: "{question}\nPlease reason step by step, and put your final answer within \\boxed{{}}.".format(**x),
    },
    "math500": {
        "load_args": ("HuggingFaceH4/MATH-500",),
        "load_kwargs": {"split": "test"},
        "format": lambda x: "{problem}\nPlease reason step by step, and put your final answer within \\boxed{{}}.".format(**x),
    },
    "humaneval": {
        "load_args": ("openai/openai_humaneval",),
        "load_kwargs": {"split": "test"},
        "format": lambda x: "Write a solution to the following problem and make sure that it passes the tests:\n```python\n{prompt}\n```".format(**x),
    },
    "mbpp": {
        "load_args": ("google-research-datasets/mbpp", "sanitized"),
        "load_kwargs": {"split": "test"},
        "format": lambda x: x["prompt"],
    },
    "mt-bench": {
        "load_args": ("HuggingFaceH4/mt_bench_prompts",),
        "load_kwargs": {"split": "train"},
        "format": lambda x: x["prompt"],
        "multi_turn": True,
    },
}


def _reasoning_kwargs(reasoning: str | None, template: str | None = None) -> dict:
    if reasoning is None:
        return {}
    if reasoning in {"on", "off"}:
        if template is not None and "enable_thinking" not in template:
            raise ValueError("This model does not support --reasoning on/off")
        return {"enable_thinking": reasoning == "on"}
    if template is not None:
        for key in ("reasoning_strength", "reasoning_effort"):
            if key in template:
                return {key: reasoning}
        raise ValueError("This model supports only --reasoning on/off")
    return {
        "enable_thinking": True,
        "reasoning_effort": reasoning,
        "reasoning_strength": reasoning,
    }


def apply_chat_template(
    tokenizer,
    messages: list[dict],
    reasoning: str | None,
) -> str:
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        **_reasoning_kwargs(reasoning, str(tokenizer.chat_template or "")),
    )


def load_transformers_models(model_id: str, draft_id: str, device):
    import torch
    from transformers import (
        AutoConfig,
        AutoModelForCausalLM,
        AutoModelForImageTextToText,
        AutoTokenizer,
    )

    from .model import DFlash2DraftModel, DFlashDraftModel

    target_kwargs = {"attn_implementation": "sdpa", "dtype": torch.bfloat16}
    try:
        target = AutoModelForCausalLM.from_pretrained(model_id, **target_kwargs)
    except ValueError:
        target = AutoModelForImageTextToText.from_pretrained(model_id, **target_kwargs)
    target = target.to(device).eval()

    config = AutoConfig.from_pretrained(draft_id)
    draft_class = (
        DFlash2DraftModel
        if "DFlash2DraftModel" in (config.architectures or [])
        else DFlashDraftModel
    )
    draft = (
        draft_class.from_pretrained(
            draft_id,
            attn_implementation="sdpa",
            dtype=torch.bfloat16,
        )
        .to(device)
        .eval()
    )
    return target, draft, AutoTokenizer.from_pretrained(model_id)


def load_mlx_models(model_id: str, draft_id: str, draft_bits: int | None):
    import mlx.core as mx
    from mlx import nn

    from .model_mlx import load, load_draft

    model, tokenizer = load(model_id)
    draft = load_draft(draft_id)
    if draft_bits is not None:
        nn.quantize(draft, group_size=64, bits=draft_bits)
        mx.eval(draft.parameters())
    return model, draft, tokenizer


def stop_token_ids(model, tokenizer) -> list[int]:
    token_ids = model.generation_config.eos_token_id or tokenizer.eos_token_id
    return [token_ids] if isinstance(token_ids, int) else list(token_ids)


def send_openai(
    base_url: str,
    messages: list[dict],
    *,
    model: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    timeout_s: int,
    reasoning: str | None = None,
) -> dict:
    body: dict = {
        "model": model,
        "messages": messages,
        "max_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "chat_template_kwargs": _reasoning_kwargs(reasoning),
        "return_meta_info": True,
    }
    if top_k > 0:
        body["top_k"] = top_k
    response = requests.post(
        base_url.rstrip("/") + "/v1/chat/completions",
        json=body,
        timeout=timeout_s,
    )
    response.raise_for_status()
    return response.json()


def load_and_process_dataset(data_name: str) -> list[dict]:
    from datasets import load_dataset

    if data_name not in DATASETS:
        raise ValueError(f"Unknown dataset '{data_name}'. Available: {list(DATASETS.keys())}")

    cfg = DATASETS[data_name]
    dataset = load_dataset(*cfg["load_args"], **cfg["load_kwargs"])
    return [
        {"turns": cfg["format"](row) if cfg.get("multi_turn") else [cfg["format"](row)]}
        for row in dataset
    ]


def _select_dataset(
    dataset: list[dict], count: int | None, *, repeat: bool = False,
) -> list[dict]:
    order = list(range(len(dataset)))
    random.Random(42).shuffle(order)
    count = len(order) if count is None else count
    if not repeat:
        count = min(count, len(order))
    return [dataset[order[i % len(order)]] for i in range(count)]


def _make_decode_metrics(num_output_tokens: int, generation_tps: float, acceptance_lengths: list[int]) -> SimpleNamespace:
    return SimpleNamespace(
        num_output_tokens=num_output_tokens,
        time_per_output_token=1.0 / generation_tps if generation_tps > 0 else float("inf"),
        acceptance_lengths=acceptance_lengths,
    )


def _print_decode_summary(responses: list[dict[int, SimpleNamespace]], block_size: int) -> None:
    baseline_tpot = statistics.mean(r[1].time_per_output_token for r in responses)
    dflash_tpot = statistics.mean(r[block_size].time_per_output_token for r in responses)
    print(f"Baseline throughput: {1 / baseline_tpot:.2f} tok/s")
    print(f"DFlash throughput:  {1 / dflash_tpot:.2f} tok/s")
    print(f"Decoding speedup: {baseline_tpot / dflash_tpot:.2f}")

    per_request = [
        r[block_size].acceptance_lengths
        for r in responses
        if r[block_size].acceptance_lengths
    ]
    acceptance_lengths = list(chain.from_iterable(r[block_size].acceptance_lengths for r in responses))
    if not acceptance_lengths:
        print("Average Acceptance length: n/a")
        return
    mean_accept = statistics.mean(statistics.mean(x) for x in per_request)
    print(f"Average Acceptance length: {mean_accept:.2f}")

    histogram = [acceptance_lengths.count(b) / len(acceptance_lengths) for b in range(block_size + 1)]
    print(f"Acceptance length histogram: {[f'{x * 100:.1f}%' for x in histogram]}")


def _run_transformers(args: argparse.Namespace) -> None:
    import torch

    from .model import dflash_generate

    torch.manual_seed(0)
    device = torch.device("cuda:0")
    target, draft_model, tokenizer = load_transformers_models(
        args.model, args.draft, device
    )

    block_size = args.block_size if args.block_size is not None else draft_model.block_size
    dataset = load_and_process_dataset(args.dataset)

    dataset = _select_dataset(dataset, args.max_samples)

    warmup_text = apply_chat_template(
        tokenizer,
        [{"role": "user", "content": dataset[0]["turns"][0]}],
        args.reasoning,
    )
    warmup = tokenizer.encode(
        warmup_text, return_tensors="pt", add_special_tokens=False
    ).to(device)
    warmup_tokens = min(64, args.max_new_tokens)
    for bs in (1, block_size):
        dflash_generate(
            draft_model, target, warmup, warmup_tokens, None,
            args.temperature, args.top_p, args.top_k, block_size=bs,
        )

    responses = []
    for idx in tqdm(range(len(dataset))):
        instance = dataset[idx]
        messages = []
        for user_content in instance["turns"]:
            messages.append({"role": "user", "content": user_content})
            input_text = apply_chat_template(
                tokenizer, messages, args.reasoning,
            )
            input_ids = tokenizer.encode(
                input_text, return_tensors="pt", add_special_tokens=False
            ).to(target.device)

            response = {}
            for bs in [1, block_size]:
                response[bs] = dflash_generate(
                    draft_model,
                    target=target,
                    input_ids=input_ids,
                    max_new_tokens=args.max_new_tokens,
                    stop_token_ids=stop_token_ids(target, tokenizer),
                    temperature=args.temperature,
                    top_p=args.top_p,
                    top_k=args.top_k,
                    block_size=bs,
                    return_stats=True,
                )

            spec_response = response[block_size]
            generated_ids = spec_response.output_ids[0, spec_response.num_input_tokens:]
            output_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            messages.append({"role": "assistant", "content": output_text})
            responses.append(response)

    _print_decode_summary(responses, block_size)


def _run_mlx(args: argparse.Namespace) -> None:
    import mlx.core as mx
    from mlx_lm import stream_generate as stream_generate_baseline

    from .model_mlx import make_sampler, stream_generate

    mx.random.seed(0)
    sampler = make_sampler(args.temperature, args.top_p, args.top_k)

    print(f"Loading target: {args.model}")
    print(f"Loading draft: {args.draft}")
    model, draft, tokenizer = load_mlx_models(
        args.model, args.draft, args.draft_bits
    )
    block_size = args.block_size if args.block_size is not None else int(draft.config.block_size)

    dataset = load_and_process_dataset(args.dataset)
    dataset = _select_dataset(dataset, args.max_samples)

    warmup_prompt = tokenizer.encode("Hi")
    list(stream_generate_baseline(model, tokenizer, warmup_prompt, 3, sampler=sampler))
    list(stream_generate(
        model, draft, tokenizer, warmup_prompt,
        block_size=block_size,
        max_tokens=3,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
    ))

    responses = []
    for idx in tqdm(range(len(dataset))):
        instance = dataset[idx]
        messages = []
        for user_content in instance["turns"]:
            messages.append({"role": "user", "content": user_content})
            prompt = apply_chat_template(
                tokenizer, messages, args.reasoning,
            )

            response = {}

            tokens_bl, tps_bl = [], 0
            for r in stream_generate_baseline(model, tokenizer, prompt, args.max_new_tokens, sampler=sampler):
                tokens_bl.append(r.token)
                tps_bl = r.generation_tps
            response[1] = _make_decode_metrics(len(tokens_bl), tps_bl, [1])

            tokens_df, accs, tps_df = [], [], 0
            for r in stream_generate(
                model, draft, tokenizer, prompt,
                block_size=block_size,
                max_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
            ):
                tokens_df.extend(r.tokens)
                if r.accepted is not None:
                    accs.append(r.accepted)
                tps_df = r.generation_tps
            response[block_size] = _make_decode_metrics(len(tokens_df), tps_df, accs)

            output_text = tokenizer.decode(tokens_df)
            messages.append({"role": "assistant", "content": output_text})
            responses.append(response)

    _print_decode_summary(responses, block_size)


def _run_openai(args: argparse.Namespace) -> None:
    bs = max(args.concurrency, 1)
    warmup_n = 0 if args.num_warmup == 0 else max(args.num_warmup, bs)
    dataset = _select_dataset(
        load_and_process_dataset(args.dataset), args.num_prompts + warmup_n, repeat=True,
    )
    prompts = [
        [{"role": "user", "content": item["turns"][0]}]
        for item in dataset[:args.num_prompts]
    ]
    warmup_prompts = [
        [{"role": "user", "content": item["turns"][0]}]
        for item in dataset[args.num_prompts:]
    ]

    def send_one(messages: list[dict], max_new_tokens=args.max_new_tokens) -> dict:
        return send_openai(
            args.base_url,
            messages,
            model=args.model,
            max_new_tokens=max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            timeout_s=args.timeout_s,
            reasoning=args.reasoning,
        )

    if warmup_n > 0:
        print(f"[warmup] {warmup_n} requests ...")
        with ThreadPoolExecutor(max_workers=bs) as pool:
            list(pool.map(lambda p: send_one(p, min(64, args.max_new_tokens)), warmup_prompts))

    print(f"Running benchmark: {args.num_prompts} prompts, concurrency={bs} ...")
    start = time.perf_counter()
    total_tokens = 0
    spec_verify_ct_sum = 0
    spec_accept_lengths: list[float] = []

    with ThreadPoolExecutor(max_workers=bs) as pool:
        futures = [pool.submit(send_one, p) for p in prompts]
        for fut in tqdm(as_completed(futures), total=len(prompts), desc="Benchmarking"):
            out = fut.result()
            usage = out.get("usage", {}) or {}
            total_tokens += int(usage.get("completion_tokens", 0))
            meta = out.get("meta_info", {}) or {}
            spec_verify_ct_sum += int(meta.get("spec_verify_ct", 0))
            if "spec_accept_length" in meta:
                try:
                    spec_accept_lengths.append(float(meta["spec_accept_length"]))
                except (TypeError, ValueError):
                    pass

    latency = time.perf_counter() - start
    toks_per_s = total_tokens / max(latency, 1e-6)

    print(f"\n{'=' * 50}")
    print(f"Backend:          {args.backend}")
    print(f"Dataset:          {args.dataset}")
    print(f"Num prompts:      {args.num_prompts}")
    print(f"Concurrency:      {bs}")
    print(f"Latency:          {latency:.1f}s")
    print(f"Output tokens:    {total_tokens}")
    print(f"Throughput:       {toks_per_s:,.2f} tok/s")
    if spec_accept_lengths:
        print(f"Accept length:    {statistics.mean(spec_accept_lengths):.3f}")
    if spec_verify_ct_sum > 0:
        print(f"Spec verify ct:   {spec_verify_ct_sum}")
    print(f"{'=' * 50}")


def run(args: argparse.Namespace) -> None:
    if args.backend == "transformers":
        _run_transformers(args)
    elif args.backend == "mlx":
        _run_mlx(args)
    else:
        _run_openai(args)
