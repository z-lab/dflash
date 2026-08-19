# DFlash：用於 Flash 投機解碼的區塊擴散

[English](README.md) | [繁體中文](README.zh-TW.md)

**DFlash** 是專為投機解碼設計的輕量級**區塊擴散**模型，可實現高效且高品質的並行草稿生成。

<details open>
<summary><strong>DFlash 2</strong></summary>

[**部落格**](https://inco.ai/blog/dflash2/) | [**模型**](https://huggingface.co/collections/z-lab/dflash-2)

<p align="center"><img src="https://raw.githubusercontent.com/jianc99/jianc99.github.io/master/images/dflash2_system.png" alt="DFlash 2 架構"></p>

https://github.com/user-attachments/assets/f786e7c5-c2bc-47d4-8a32-1f730a689e1b
</details>

<details>
<summary><strong>DFlash</strong></summary>

[**論文**](https://arxiv.org/abs/2602.06036) | [**部落格**](https://z-lab.ai/projects/dflash/) | [**模型**](https://huggingface.co/collections/z-lab/dflash)

![DFlash 架構](https://raw.githubusercontent.com/jianc99/jianc99.github.io/master/images/dflash_system.png)

https://github.com/user-attachments/assets/5b29cabb-eb95-44c9-8ffe-367c0758de8c
</details>

## 支援的模型

### DFlash 2

目前可用的檢查點：[Muse-Glimmer-30B](https://huggingface.co/z-lab/Muse-Glimmer-30B-DFlash2) 與 [Qwen3.8-27B](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2)。最新更新請見 [DFlash 2 合集](https://huggingface.co/collections/z-lab/dflash-2)。

### DFlash

公開檢查點位於 [DFlash 合集](https://huggingface.co/collections/z-lab/dflash)：

- **Qwen:** Qwen3.6 (27B, 35B-A3B), Qwen3.5 (4B, 9B, 27B, 35B-A3B, 122B-A10B, 397B-A17B), Qwen3 (4B/8B 非思考模式, Coder-Next, Coder-30B-A3B)
- **Gemma:** Gemma 4 (12B, 31B, 26B-A4B)
- **MiniMax:** M2.5, M2.7
- **Kimi:** K2.5, K2.6, K2.7-Code
- **其他:** GPT-OSS (20B, 120B), Llama-3.1-8B, GLM 5.1, Alpamayo 1.5/R1 10B

下方的 Transformers 與 MLX 後端僅支援其明確列出的模型系列。其他檢查點可透過 OpenAI 相容的 SGLang 或 vLLM 伺服器進行基準測試。

## 📦 安裝

安裝基礎套件即可搭配 OpenAI 相容伺服器使用，或一併安裝本機推理相依套件。本機安裝在 Apple Silicon 上使用 MLX，在 Linux 上使用 Transformers。

```bash
pip install dflash
pip install "dflash[local]"  # 本機推理
```

若要進行服務端基準測試，請另行安裝支援版本的 [SGLang](https://github.com/sgl-project/sglang/pull/35371)、[vLLM](https://github.com/vllm-project/vllm/pull/52816)、[oMLX](https://github.com/z-lab/omlx-fork/releases/download/0.6.2-dflash2/oMLX-0.6.2-zlab-dflash2-arm64-signed.dmg) 或 [llama.cpp](https://github.com/ggml-org/llama.cpp/pull/27342)，啟動其具備 DFlash 的 OpenAI 相容伺服器，並在下方傳入其 `--base-url`。

## 🚀 快速開始

### Transformers

Transformers 後端支援 Muse-Glimmer-30B 的 DFlash 2，以及 Qwen3 與 LLaMA-3.1-8B 的 DFlash。Muse 使用 `reasoning_strength`：`low`、`medium`、`high`（預設）或 `xhigh`。

```bash
dflash generate transformers \
    --model meta-models/Muse-Glimmer-30B \
    --draft z-lab/Muse-Glimmer-30B-DFlash2 \
    --reasoning high --temperature 1 --top-p 0.95 --top-k 64 \
    "How many positive whole-number divisors does 196 have?"
```

### MLX（Apple Silicon）

MLX 後端支援 Qwen3.8-27B 的 DFlash 2，以及 Qwen3、Qwen3.5、Qwen3.6 與 Gemma 4 的 DFlash。Qwen3.8 使用 `reasoning_effort`：`low`、`medium` 或 `xhigh`（預設）。若目標模型或草稿模型經過量化，請使用 `block_size <= 5`：MLX 目前的量化矩陣乘法核心在較大驗證寬度下效率會下降。以下範例以 4-bit 權重同時執行目標模型與草稿模型。

```bash
dflash generate mlx \
    --model mlx-community/Qwen3.8-27B-4bit \
    --draft z-lab/Qwen3.8-27B-DFlash2 \
    --draft-bits 4 --block-size 5 --reasoning xhigh \
    "How many positive whole-number divisors does 196 have?"
```

### OpenAI 相容伺服器

請另行啟動最新的 SGLang 或 vLLM 伺服器，然後執行：

```bash
dflash generate openai \
    --base-url http://127.0.0.1:8000 --model Qwen/Qwen3.8-27B \
    "How many positive whole-number divisors does 196 have?"
```

## 📊 評估

所有基準測試使用相同的資料集（gsm8k、math500、humaneval、mbpp、mt-bench），由 Hugging Face Datasets 下載並快取。

**OpenAI 相容伺服器**（SGLang 或 vLLM）：
```bash
dflash benchmark openai \
    --base-url http://127.0.0.1:8000 --model Qwen/Qwen3.8-27B \
    --dataset gsm8k --num-prompts 128 --concurrency 1 --reasoning xhigh \
    --temperature 1 --top-p 0.95 --top-k 20
```

**Transformers**（Muse-Glimmer-30B DFlash 2）：
```bash
dflash benchmark transformers \
    --model meta-models/Muse-Glimmer-30B --draft z-lab/Muse-Glimmer-30B-DFlash2 \
    --dataset gsm8k --max-samples 128 --reasoning high
```

**MLX**（Qwen3.8-27B 4-bit DFlash 2）：
```bash
dflash benchmark mlx \
    --model mlx-community/Qwen3.8-27B-4bit --draft z-lab/Qwen3.8-27B-DFlash2 \
    --dataset gsm8k --max-samples 128 --reasoning xhigh --block-size 5 --draft-bits 4
```

## 致謝

非常感謝 [@dcw02](https://github.com/dcw02)、[@gongy](https://github.com/gongy) 以及 [@modal-labs](https://github.com/modal-labs) 團隊，以快速且高品質的支援將 DFlash 引入 SGLang。同時也非常感謝 NVIDIA 的 [@benchislett](https://github.com/benchislett)，將 DFlash 帶入 vLLM，並讓更廣泛的推理服務社群得以使用。

## 引用

若你覺得 DFlash 有幫助，請引用我們的工作。如要分享對 DFlash 的回饋或請求支援新模型，請填寫此表單：[DFlash 回饋](https://forms.gle/4YNwfqb4nJdqn6hq9)。

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
