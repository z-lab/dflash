# DFlash: Block Diffusion for Flash Speculative Decoding
[**Paper**](https://arxiv.org/abs/2602.06036) | [**Blog**](https://z-lab.ai/projects/dflash/) | [**Models**](https://huggingface.co/collections/z-lab/dflash)

**DFlash** is a lightweight **block diffusion** model designed for speculative decoding. It enables efficient and high-quality parallel drafting.
<br>

<div align="center">
  <img src="assets/dflash_system.png" alt="DFlash Architecture" width="100%">
</div>

https://github.com/user-attachments/assets/5b29cabb-eb95-44c9-8ffe-367c0758de8c

<br>

## 📦 Model Support Plan

### ✅ Supported
- **Qwen3-4B**: https://huggingface.co/z-lab/Qwen3-4B-DFlash-b16  
- **Qwen3-8B**: https://huggingface.co/z-lab/Qwen3-8B-DFlash-b16  
- **Qwen3-Coder-30B-A3B**: https://huggingface.co/z-lab/Qwen3-Coder-30B-A3B-DFlash
- **Llama-3.1-8B-Instruct**: https://huggingface.co/z-lab/LLaMA3.1-8B-Instruct-DFlash-UltraChat
- **openai/gpt-oss-20b**: https://huggingface.co/z-lab/gpt-oss-20b-DFlash

### 🚧 Coming Soon
- **Qwen/Qwen3-Coder-Next** (Very soon)
- **openai/gpt-oss-120b**  
- **zai-org/GLM-4.7**
- **zai-org/GLM-4.7-Flash**

> 💡 Feel free to open a GitHub issue if you’d like to request support for additional models!  
> We will also open-source the training recipe soon, so you can train your own DFlash draft model to accelerate any LLM.

<br>

## 🚀 Quick Start

### Installation
```bash
conda create -n dflash python=3.11
conda activate dflash

git clone https://github.com/z-lab/dflash.git
cd dflash

pip install uv
uv pip install -r requirements.txt

# Optionally install flash-attn.
# If unavailable, evaluation falls back to torch.sdpa in the Transformers backend.
# The measured speedup will be slower, but the acceptance length remains comparable.

# uv pip install flash-attn --no-build-isolation
```

### SGLang

```bash
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1

python -m sglang.launch_server \
    --model-path Qwen/Qwen3-Coder-30B-A3B-Instruct \
    --speculative-algorithm DFLASH \
    --speculative-draft-model-path z-lab/Qwen3-Coder-30B-A3B-DFlash \
    --tp-size 1 \
    --dtype bfloat16 \
    --attention-backend fa3 \
    --mem-fraction-static 0.75 \
    --trust-remote-code
```

### Transformers

```python
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

model = AutoModel.from_pretrained(
    "z-lab/Qwen3-8B-DFlash-b16", 
    trust_remote_code=True, 
    dtype="auto", 
    device_map="cuda:0"
).eval()

target = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-8B", 
    dtype="auto", 
    device_map="cuda:0"
).eval()

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
prompt = "How many positive whole-number divisors does 196 have?"
messages = [
    {"role": "user", "content": prompt}
]
# Note: this draft model is used for thinking mode disabled
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

generate_ids = model.spec_generate(
    input_ids=model_inputs["input_ids"], 
    max_new_tokens=2048, 
    temperature=0.0, 
    target=target, 
    stop_token_ids=[tokenizer.eos_token_id]
)

print(tokenizer.decode(generate_ids[0], skip_special_tokens=False))
```

## 📊 Evaluation
We provide scripts to reproduce the speedup and acceptance length metrics in the paper. The reported results were tested on NVIDIA H200 or B200 GPUs.

To run benchmark on Transformers backend:
```bash
bash run_benchmark.sh
```

To run benchmark on SGLang:
```bash
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1

python benchmark_sglang.py \
  --target-model Qwen/Qwen3-8B \
  --draft-model z-lab/Qwen3-8B-DFlash-b16 \
  --concurrencies 1,4,8,16,32 \
  --dataset-name math500 \
  --attention-backends fa3,flashinfer \
  --tp-size 1 \
  --output-md sglang_results.md
```

<div align="center">
  <img src="assets/dflash_results.png" width="100%">
</div>

## **Acknowledgement**

Huge thanks to [@dcw02](https://github.com/dcw02), [@gongy](https://github.com/gongy), and the other folks at [@modal-labs](https://github.com/modal-labs) for the fast, high-quality support in bringing DFlash into SGLang—making it possible to truly accelerate LLM serving in real-world deployments.

## **Citation**
If you find DFlash useful for your research or applications, please cite our project.

```bibtex
@misc{chen2026dflash,
  title         = {DFlash: Block Diffusion for Flash Speculative Decoding},
  author        = {Chen, Jian and Liang, Yesheng and Liu, Zhijian},
  year          = {2026},
  eprint        = {2602.06036},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  url           = {https://arxiv.org/abs/2602.06036}
}
```
