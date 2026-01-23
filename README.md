# DFlash: Block Diffusion for Flash Speculative Decoding
[**Paper (Coming Soon)**](#) | [**Blog**](https://z-lab.ai/projects/dflash/) | [**Models**](https://huggingface.co/collections/z-lab/dflash)

**DFlash** is a lightweight **block diffusion** model designed for speculative decoding. It enables efficient and high-quality parallel drafting.
<br>

<div align="center">
  <img src="assets/dflash_system.png" alt="DFlash Architecture" width="100%">
</div>

https://github.com/user-attachments/assets/5b29cabb-eb95-44c9-8ffe-367c0758de8c

<br>

## 🚀 Quick Start

### Serving with SGLang

**DFlash is now supported on SGLang**, enabling high-throughput speculative decoding in a production-grade serving stack.  
**vLLM integration is currently in progress.**

#### Installation
```bash
uv pip install "git+https://github.com/sgl-project/sglang.git@refs/pull/16818/head#subdirectory=python"
```

#### Serving
```bash
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

#### Installation
```bash
conda create -n dflash python=3.11
conda activate dflash

git clone https://github.com/z-lab/dflash.git
cd dflash

pip install -r requirements.txt
pip install flash-attn --no-build-isolation
```

#### Example Usage
The following example demonstrates how to load the DFlash drafter and the Qwen3-8B target model to perform speculative decoding.
```python
import torch
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

# 1. Load the DFlash Draft Model
# Note: trust_remote_code=True is required for the custom diffusion architecture. We recommend run on one GPU currently.
model = AutoModel.from_pretrained(
    "z-lab/Qwen3-8B-DFlash-b16", 
    trust_remote_code=True, 
    dtype="auto", 
    device_map="cuda:0"
).eval()

# 2. Load the Target Model
target = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-8B", 
    dtype="auto", 
    device_map="cuda:0"
).eval()

# 3. Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
# Essential: Add the mask token required for diffusion steps
tokenizer.add_special_tokens({"mask_token": "<|MASK|>"})

# 4. Prepare Input
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

# 5. Run Speculative Decoding
# The 'spec_generate' function is a custom method provided by the DFlash model
generate_ids = model.spec_generate(
    input_ids=model_inputs["input_ids"], 
    max_new_tokens=2048, 
    temperature=0.0, 
    target=target, 
    mask_token_id=tokenizer.mask_token_id, 
    stop_token_ids=[tokenizer.eos_token_id]
)

print(tokenizer.decode(generate_ids[0], skip_special_tokens=True))
```

## 📊 Evaluation & Benchmarks
We provide scripts to reproduce our speedup and acceptance length metrics. The reported results were tested on NVIDIA B200 GPUs.

To run the benchmark:
```bash
bash run_benchmark.sh
```

<div align="center">
  <img src="assets/dflash_results.png" width="100%">
</div>

## **Citation**
If you find DFlash useful for your research or applications, please cite our project. The full paper is coming soon!

```bibtex
@article{chen2026dflash,
  title   = {DFlash: Block Diffusion for Flash Speculative Decoding},
  author  = {Chen, Jian and Liu, Zhijian},
  journal = {arXiv preprint},
  year    = {2026},
  url     = {[https://github.com/z-lab/dflash](https://github.com/z-lab/dflash)},
  note    = {Paper coming soon}
}
