# tester.py
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import PeftModel
from seal.evaluate import eval_knowledge

# 1) Define your 4-bit config exactly as in training
bnb_cfg = BitsAndBytesConfig(
    load_in_4bit              = True,
    bnb_4bit_use_double_quant = True,
    bnb_4bit_compute_dtype    = torch.float16,
)

# 2) Load the *base* model in 4-bit
base = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3-8B-Instruct",
    device_map="auto",
    quantization_config=bnb_cfg,
    local_files_only=True,      # force local-only so it won't hit HF hub
)

# 3) Wrap it in your LoRA adapter
model = PeftModel.from_pretrained(
    base,
    "./ckpt_run1/policy",       # path where you saved adapter weights
    device_map="auto",
    local_files_only=True,
)

# 4) Load the finetuned tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    "./ckpt_run1/tokenizer",
    use_fast=True,
    local_files_only=True,
)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

# 5) Prepare your eval set exactly as in conf/knowledge.yaml
eval_ds = load_dataset("squad", split="validation[:2000]")

# 6) Build the minimal list-of-dicts for eval_knowledge
eval_examples = [
    {"question": ex["question"], "ground_truth": ex["answers"]["text"][0]}
    for ex in eval_ds
]

# 7) Run evaluation
f1, em = eval_knowledge(model, tokenizer, eval_examples)
print(f"→ Final eval on 2000 examples: F1 = {f1:.3f}, EM = {em:.3f}")

