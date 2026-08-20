# DFlash：用于 Flash 投机解码的块扩散模型

[**论文**](https://arxiv.org/abs/2602.06036) | [**博客**](https://z-lab.ai/projects/dflash/) | [**模型**](https://huggingface.co/collections/z-lab/dflash)

**DFlash** 是一个轻量级的**块扩散**模型，专为投机解码设计。它能够实现高效、高质量的并行草稿生成。

![DFlash 架构](https://raw.githubusercontent.com/jianc99/jianc99.github.io/master/images/dflash_system.png)

https://github.com/user-attachments/assets/5b29cabb-eb95-44c9-8ffe-367c0758de8c

## 支持的模型

| 模型 | DFlash 草稿模型 |
|---|---|
| gemma-4-26B-A4B-it | [z-lab/gemma-4-26B-A4B-it-DFlash](https://huggingface.co/z-lab/gemma-4-26B-A4B-it-DFlash) |
| gemma-4-31B-it | [z-lab/gemma-4-31B-it-DFlash](https://huggingface.co/z-lab/gemma-4-31B-it-DFlash) |
| Qwen3.6-27B | [z-lab/Qwen3.6-27B-DFlash](https://huggingface.co/z-lab/Qwen3.6-27B-DFlash) |
| Qwen3.6-35B-A3B | [z-lab/Qwen3.6-35B-A3B-DFlash](https://huggingface.co/z-lab/Qwen3.6-35B-A3B-DFlash) |
| MiniMax-M2.5 (预览版) | [z-lab/MiniMax-M2.5-DFlash](https://huggingface.co/z-lab/MiniMax-M2.5-DFlash) |
| Kimi-K2.5 | [z-lab/Kimi-K2.5-DFlash](https://huggingface.co/z-lab/Kimi-K2.5-DFlash) |
| Qwen3.5-4B | [z-lab/Qwen3.5-4B-DFlash](https://huggingface.co/z-lab/Qwen3.5-4B-DFlash) |
| Qwen3.5-9B | [z-lab/Qwen3.5-9B-DFlash](https://huggingface.co/z-lab/Qwen3.5-9B-DFlash) |
| Qwen3.5-27B | [z-lab/Qwen3.5-27B-DFlash](https://huggingface.co/z-lab/Qwen3.5-27B-DFlash) |
| Qwen3.5-35B-A3B | [z-lab/Qwen3.5-35B-A3B-DFlash](https://huggingface.co/z-lab/Qwen3.5-35B-A3B-DFlash) |
| Qwen3.5-122B-A10B | [z-lab/Qwen3.5-122B-A10B-DFlash](https://huggingface.co/z-lab/Qwen3.5-122B-A10B-DFlash) |
| Qwen3-Coder-Next | [z-lab/Qwen3-Coder-Next-DFlash](https://huggingface.co/z-lab/Qwen3-Coder-Next-DFlash) |
| Qwen3-Coder-30B-A3B | [z-lab/Qwen3-Coder-30B-A3B-DFlash](https://huggingface.co/z-lab/Qwen3-Coder-30B-A3B-DFlash) |
| gpt-oss-20b | [z-lab/gpt-oss-20b-DFlash](https://huggingface.co/z-lab/gpt-oss-20b-DFlash) |
| gpt-oss-120b | [z-lab/gpt-oss-120b-DFlash](https://huggingface.co/z-lab/gpt-oss-120b-DFlash) |
| Qwen3-4B (非思考模式) | [z-lab/Qwen3-4B-DFlash-b16](https://huggingface.co/z-lab/Qwen3-4B-DFlash-b16) |
| Qwen3-8B (非思考模式) | [z-lab/Qwen3-8B-DFlash-b16](https://huggingface.co/z-lab/Qwen3-8B-DFlash-b16) |
| Llama-3.1-8B-Instruct | [z-lab/LLaMA3.1-8B-Instruct-DFlash-UltraChat](https://huggingface.co/z-lab/LLaMA3.1-8B-Instruct-DFlash-UltraChat) |
| DeepSeek-V4-Flash | 即将推出 |
| DeepSeek-V4-Pro | 即将推出 |
| MiniMax-M2.7 | 即将推出 |
| GLM-5.1 | 即将推出 |

> 欢迎提交 GitHub Issue 请求支持更多模型。我们也将很快开源训练配方，您可以根据自己的需求训练 DFlash 草稿模型来加速任何 LLM。

## 📦 安装

建议使用独立的虚拟环境以避免冲突。

| 后端 | 安装命令 |
|---|---|
| **Transformers** | `uv pip install -e ".[transformers]"` |
| **SGLang** | `uv pip install -e ".[sglang]"` |
| **vLLM** | 详见下文 |
| **MLX** (Apple Silicon) | `pip install -e ".[mlx]"` |

**vLLM：** vLLM v0.20.1+ 已包含 DFlash 核心支持。对于大多数模型使用标准安装：
```bash
uv pip install -e ".[vllm]"
```

Gemma4 DFlash 目前需要我们的临时 vLLM Gemma4 构建版本。推荐使用 Docker：
```bash
docker pull ghcr.io/z-lab/vllm-openai:gemma4-dflash-cu130
```

Gemma4 源码备用安装：
```bash
uv pip install -U --torch-backend=auto \
  "vllm @ git+https://github.com/vllm-project/vllm.git@refs/pull/41703/head"
```

较新的非 Gemma4 SWA 草稿模型使用 SWA 支持分支：
```bash
uv pip install -U --torch-backend=auto \
  "vllm @ git+https://github.com/vllm-project/vllm.git@refs/pull/40898/head"
```

## 🚀 快速开始

### vLLM

使用 Docker 运行 Gemma4：
```bash
docker run --rm -it \
  --gpus all \
  --ipc=host \
  --shm-size=16g \
  -p 8000:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  ghcr.io/z-lab/vllm-openai:gemma4-dflash-cu130 \
  google/gemma-4-26B-A4B-it \
  --host 0.0.0.0 \
  --port 8000 \
  --speculative-config '{"method": "dflash", "model": "z-lab/gemma-4-26B-A4B-it-DFlash", "num_speculative_tokens": 15, "attention_backend": "flash_attn"}' \
  --attention-backend triton_attn \
  --max-num-batched-tokens 32768 \
  --trust-remote-code
```

非 Gemma4 模型：
```bash
vllm serve Qwen/Qwen3.5-27B \
  --speculative-config '{"method": "dflash", "model": "z-lab/Qwen3.5-27B-DFlash", "num_speculative_tokens": 15}' \
  --attention-backend flash_attn \
  --max-num-batched-tokens 32768
```

### SGLang

```bash
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1

# 可选：启用调度重叠（实验性，可能不稳定）
# export SGLANG_ENABLE_SPEC_V2=1
# export SGLANG_ENABLE_DFLASH_SPEC_V2=1
# export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1

python -m sglang.launch_server \
    --model-path Qwen/Qwen3.5-35B-A3B \
    --speculative-algorithm DFLASH \
    --speculative-draft-model-path z-lab/Qwen3.5-35B-A3B-DFlash \
    --speculative-num-draft-tokens 16 \
    --tp-size 1 \
    --attention-backend trtllm_mha \
    --speculative-draft-attention-backend fa4 \
    --mem-fraction-static 0.75 \
    --mamba-scheduler-strategy extra_buffer \
    --trust-remote-code
```

### Transformers

仅 Qwen3 和 LLaMA-3.1 模型支持 Transformers 后端。

```python
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

draft = AutoModel.from_pretrained("z-lab/Qwen3-8B-DFlash-b16", trust_remote_code=True, dtype="auto", device_map="cuda:0").eval()
target = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B", dtype="auto", device_map="cuda:0").eval()
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")

messages = [{"role": "user", "content": "How many positive whole-number divisors does 196 have?"}]
input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True, enable_thinking=False).to(draft.device)

output = draft.spec_generate(input_ids=input_ids, max_new_tokens=2048, temperature=0.0, target=target, stop_token_ids=[tokenizer.eos_token_id])
print(tokenizer.decode(output[0], skip_special_tokens=False))
```

### MLX (Apple Silicon)

社区已有许多优秀的 MLX DFlash 实现；我们在这里提供一个简单高效的版本，已在 Apple M5 Pro 上使用 Qwen3、Qwen3.5 和 Gemma-4 模型进行了测试。

```python
from dflash.model_mlx import load, load_draft, stream_generate

model, tokenizer = load("Qwen/Qwen3.5-4B")
draft = load_draft("z-lab/Qwen3.5-4B-DFlash")

messages = [{"role": "user", "content": "How many positive whole-number divisors does 196 have?"}]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=True)
tps = 0.0
for r in stream_generate(model, draft, tokenizer, prompt, block_size=16, max_tokens=2048, temperature=0.6):
    print(r.text, end="", flush=True)
    tps = r.generation_tps
print(f"\nThroughput: {tps:.2f} tok/s")
```

## 📊 评估

所有基准测试使用相同的数据集（gsm8k、math500、humaneval、mbpp、mt-bench）。数据集在首次运行时自动下载并缓存为 JSONL 文件到 `cache/` 目录。

**vLLM**：
```bash
python -m dflash.benchmark --backend vllm \
    --base-url http://127.0.0.1:8000 --model Qwen/Qwen3.5-27B \
    --dataset gsm8k --num-prompts 128 --concurrency 1 --enable-thinking
```

**SGLang**：
```bash
python -m dflash.benchmark --backend sglang \
    --base-url http://127.0.0.1:30000 --model Qwen/Qwen3.5-35B-A3B \
    --dataset gsm8k --num-prompts 128 --concurrency 1 --enable-thinking
```

**Transformers**（仅支持 Qwen3 和 LLaMA）：
```bash
torchrun --nproc_per_node=8 -m dflash.benchmark --backend transformers \
    --model Qwen/Qwen3-8B --draft-model z-lab/Qwen3-8B-DFlash-b16 \
    --dataset gsm8k --max-samples 128
```

**MLX**：
```bash
python -m dflash.benchmark --backend mlx \
    --model mlx-community/gemma-4-31b-it-4bit --draft-model z-lab/gemma-4-31B-it-DFlash \
    --dataset gsm8k --max-samples 128 --enable-thinking
```

## 致谢

非常感谢 [@dcw02](https://github.com/dcw02)、[@gongy](https://github.com/gongy) 以及 [@modal-labs](https://github.com/modal-labs) 团队在将 DFlash 引入 SGLang 方面提供的快速、高质量支持。同时也感谢 NVIDIA 的 [@benchislett](https://github.com/benchislett) 在将 DFlash 引入 vLLM 并使其惠及更广泛的推理社区方面所做的贡献。

## 引用

如果您觉得 DFlash 有用，请引用我们的工作。如需分享关于 DFlash 的反馈或请求新模型支持，请填写此表单：[DFlash 反馈](https://forms.gle/4YNwfqb4nJdqn6hq9)。

```bibtex
@article{chen2026dflash,
  title   = {{DFlash: Block Diffusion for Flash Speculative Decoding}},
  author  = {Chen, Jian and Liang, Yesheng and Liu, Zhijian},
  journal = {arXiv preprint arXiv:2602.06036},
  year    = {2026}
}
```