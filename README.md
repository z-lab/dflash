# DFlash: Block Diffusion for Flash Speculative Decoding

**DFlash** is a lightweight **block diffusion** model designed for speculative decoding. It enables efficient and high-quality parallel drafting.

<details open>
<summary><strong>DFlash 2</strong></summary>

[**Blog**](https://inco.ai/blog/dflash2/) | [**Models**](https://huggingface.co/collections/z-lab/dflash-2)

<p align="center"><img src="https://raw.githubusercontent.com/jianc99/jianc99.github.io/master/images/dflash2_system.png" alt="DFlash 2 architecture"></p>

https://github.com/user-attachments/assets/f786e7c5-c2bc-47d4-8a32-1f730a689e1b
</details>

<details>
<summary><strong>DFlash</strong></summary>

[**Paper**](https://arxiv.org/abs/2602.06036) | [**Blog**](https://z-lab.ai/projects/dflash/) | [**Models**](https://huggingface.co/collections/z-lab/dflash)

![DFlash architecture](https://raw.githubusercontent.com/jianc99/jianc99.github.io/master/images/dflash_system.png)

https://github.com/user-attachments/assets/5b29cabb-eb95-44c9-8ffe-367c0758de8c
</details>

## Supported Models

### DFlash 2

Available checkpoints: [Muse-Glimmer-30B](https://huggingface.co/z-lab/Muse-Glimmer-30B-DFlash2) and [Qwen3.8-27B](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2). See the [DFlash 2 collection](https://huggingface.co/collections/z-lab/dflash-2) for updates.

### DFlash

Public checkpoints are available in the [DFlash collection](https://huggingface.co/collections/z-lab/dflash):

- **Qwen:** Qwen3.6 (27B, 35B-A3B), Qwen3.5 (4B, 9B, 27B, 35B-A3B, 122B-A10B, 397B-A17B), Qwen3 (4B/8B non-thinking, Coder-Next, Coder-30B-A3B)
- **Gemma:** Gemma 4 (12B, 31B, 26B-A4B)
- **MiniMax:** M2.5, M2.7
- **Kimi:** K2.5, K2.6, K2.7-Code
- **Others:** GPT-OSS (20B, 120B), Llama-3.1-8B, GLM 5.1, Alpamayo 1.5/R1 10B

Use the Transformers or MLX backends below for their explicitly listed model families. Other checkpoints can be benchmarked through an OpenAI-compatible SGLang or vLLM server.

## 📦 Installation

Install the base package for an OpenAI-compatible server, or include local
inference dependencies. The local install uses MLX on Apple Silicon and
Transformers on Linux.

```bash
pip install dflash
pip install "dflash[local]"  # local inference
```

For serving benchmarks, install a supported version of [SGLang](https://github.com/sgl-project/sglang/pull/35371), [vLLM](https://github.com/vllm-project/vllm/pull/52816), [oMLX](https://github.com/z-lab/omlx-fork/releases/download/0.6.2-dflash2/oMLX-0.6.2-zlab-dflash2-arm64-signed.dmg), or [llama.cpp](https://github.com/ggml-org/llama.cpp/pull/27342) separately, launch its OpenAI-compatible server with DFlash, and pass its `--base-url` below.

## 🚀 Quick Start

### Transformers

The Transformers backend supports DFlash 2 for Muse-Glimmer-30B, and DFlash for
Qwen3 and LLaMA-3.1-8B. Muse uses `reasoning_strength`: `low`, `medium`, `high`
(default), or `xhigh`.

```bash
dflash generate transformers \
    --model meta-models/Muse-Glimmer-30B \
    --draft z-lab/Muse-Glimmer-30B-DFlash2 \
    --reasoning high --temperature 1 --top-p 0.95 --top-k 64 \
    "How many positive whole-number divisors does 196 have?"
```

### MLX (Apple Silicon)

The MLX backend supports DFlash 2 for Qwen3.8-27B, and DFlash for Qwen3,
Qwen3.5, Qwen3.6, and Gemma 4. Qwen3.8 uses `reasoning_effort`: `low`, `medium`,
or `xhigh` (default). For quantized targets or drafts, use `block_size <= 5`: MLX's current
quantized matmul kernel becomes less efficient at larger verify widths.
The example below runs both the target and draft with 4-bit weights.

```bash
dflash generate mlx \
    --model mlx-community/Qwen3.8-27B-4bit \
    --draft z-lab/Qwen3.8-27B-DFlash2 \
    --draft-bits 4 --block-size 5 --reasoning xhigh \
    "How many positive whole-number divisors does 196 have?"
```

### OpenAI-compatible server

Launch the latest SGLang or vLLM server separately, then run:

```bash
dflash generate openai \
    --base-url http://127.0.0.1:8000 --model Qwen/Qwen3.8-27B \
    "How many positive whole-number divisors does 196 have?"
```

#### NVIDIA Jetson AGX Thor

Server startup and generation with Qwen3.8-27B DFlash 2 were tested on a
128 GB Jetson AGX Thor Developer Kit (`sm_110a`, Ubuntu 24.04, Linux aarch64)
with vLLM commit `b389ac29465b33f9e9c534df221ea3c129e9793f`, NVIDIA PyTorch
`2.13.0a0+9186a08b2c.nv26.07`, FlashInfer `0.6.17`, and CUDA 13.3-built
kernels. The target used compressed-tensors NVFP4 weights and an FP8 E4M3 KV
cache. The tested run used explicit V2 and FlashInfer architecture overrides:

```bash
export VLLM_USE_V2_MODEL_RUNNER=1
export FLASHINFER_CUDA_ARCH_LIST=11.0a

vllm serve /path/to/Qwen3.8-27B-NVFP4 \
    --served-model-name Qwen/Qwen3.8-27B \
    --max-model-len 262144 \
    --kv-cache-dtype fp8 \
    --gpu-memory-utilization 0.30 \
    --kv-cache-memory-bytes 21474836480 \
    --speculative-config \
      '{"method":"dflash","model":"z-lab/Qwen3.8-27B-DFlash2","num_speculative_tokens":7}'
```

The explicit 20 GiB allocation makes vLLM skip automatic KV-cache memory
sizing; model warmup and compilation still run. In the tested vLLM revision,
`--gpu-memory-utilization 0.30` remained relevant to the startup free-memory
check but did not size the KV cache. With the tested target, vLLM reported
453,669 tokens of KV capacity (1.73 sequences at the 262,144-token limit).
Capacity depends on the target architecture and KV-cache dtype, so verify the
reported cache size and tune both memory values for the workloads sharing the
device.

## 📊 Evaluation

All benchmarks share the same datasets (gsm8k, math500, humaneval, mbpp, mt-bench), downloaded and cached by Hugging Face Datasets.

**OpenAI-compatible server** (SGLang or vLLM):
```bash
dflash benchmark openai \
    --base-url http://127.0.0.1:8000 --model Qwen/Qwen3.8-27B \
    --dataset gsm8k --num-prompts 128 --concurrency 1 --reasoning xhigh \
    --temperature 1 --top-p 0.95 --top-k 20
```

**Transformers** (Muse-Glimmer-30B DFlash 2):
```bash
dflash benchmark transformers \
    --model meta-models/Muse-Glimmer-30B --draft z-lab/Muse-Glimmer-30B-DFlash2 \
    --dataset gsm8k --max-samples 128 --reasoning high
```

**MLX** (Qwen3.8-27B 4-bit DFlash 2):
```bash
dflash benchmark mlx \
    --model mlx-community/Qwen3.8-27B-4bit --draft z-lab/Qwen3.8-27B-DFlash2 \
    --dataset gsm8k --max-samples 128 --reasoning xhigh --block-size 5 --draft-bits 4
```

## Acknowledgement

Huge thanks to [@dcw02](https://github.com/dcw02), [@gongy](https://github.com/gongy), and the team at [@modal-labs](https://github.com/modal-labs) for their fast, high-quality support in bringing DFlash to SGLang. And huge thanks as well to [@benchislett](https://github.com/benchislett) at NVIDIA for his work in bringing DFlash to vLLM and helping make it available to the broader serving community.

## Citation
If you find DFlash useful, please cite our work. To share feedback on DFlash or request new model support, please fill out this form: [DFlash Feedback](https://forms.gle/4YNwfqb4nJdqn6hq9).

```bibtex
@article{chen2026dflash,
  title   = {{DFlash: Block Diffusion for Flash Speculative Decoding}},
  author  = {Chen, Jian and Liang, Yesheng and Liu, Zhijian},
  journal = {arXiv preprint arXiv:2602.06036},
  year    = {2026}
}

@misc{inco2026dflash2,
  title  = {DFlash 2: Keep Drafting Parallel},
  author = {{Inco AI}},
  year   = {2026},
  month  = {August},
  url    = {https://inco.ai/blog/dflash2/}
}
```
