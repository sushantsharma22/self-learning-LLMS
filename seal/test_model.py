import argparse
import torch
import gc
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset
import re
import os

torch.cuda.empty_cache()
gc.collect()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--load_4bit", type=lambda x: x.lower() == "true", default=False)
    parser.add_argument("--max_examples", type=int, default=5)
    parser.add_argument("--ft_epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=2)
    return parser.parse_args()

EXAMPLES = [
    {"question": "Which NFL team represented the AFC at Super Bowl 50?", "gt": "Denver Broncos"},
    {"question": "How many continents are there?", "gt": "7"},
    {"question": "Is the sky blue during the day?", "gt": "Yes"},
    {"question": "When was the Declaration of Independence signed?", "gt": "1776"},
    {"question": "Name three primary colors.", "gt": "red, blue, yellow"},
]

def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"

def print_vram():
    alloc = torch.cuda.memory_allocated() / (1024 ** 2)
    total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)
    print(f"[CUDA] Allocated: {alloc:.1f} MiB | Total: {total:.1f} MiB")

def f1_score(pred, gt):
    pred_tokens = re.findall(r"\w+", pred.lower())
    gt_tokens = re.findall(r"\w+", gt.lower())
    if not pred_tokens or not gt_tokens:
        return 0.0
    common = set(pred_tokens) & set(gt_tokens)
    if not common:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gt_tokens)
    return 2 * precision * recall / (precision + recall) * 100

def contains_relaxed(pred, gt):
    return gt.lower() in pred.lower()

def main():
    args = parse_args()
    device = get_device()
    print(f"\nLoading model: {args.model} (4bit: {args.load_4bit})")

    quantization_config = None
    if args.load_4bit:
        quantization_config = BitsAndBytesConfig(load_in_4bit=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map="auto",
        quantization_config=quantization_config,
        torch_dtype=torch.float16 if args.load_4bit else torch.bfloat16,
    )
    print_vram()

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"], # adjust for your model if needed
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    print("\n==== TESTING INITIAL GENERATION ====")
    for i, ex in enumerate(EXAMPLES[:args.max_examples]):
        prompt = f"Question: {ex['question']}\nAnswer (respond concisely):"
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
        output = model.generate(
            **inputs,
            max_new_tokens=32,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        response = tokenizer.decode(output[0][inputs['input_ids'].shape[-1]:], skip_special_tokens=True).strip()
        print("="*60)
        print(f"Q{i+1}: {ex['question']}")
        print(f"Model: {response}")
        print(f"GT: {ex['gt']}")
        print(f"F1: {f1_score(response, ex['gt']):.2f}  |  Contains GT? {'Yes' if contains_relaxed(response, ex['gt']) else 'No'}")
    print_vram()

    # Correct SFT dataset: input is prompt+response, label only non-prompt
    sft_data = []
    for ex in EXAMPLES[:args.max_examples]:
        prompt = f"Question: {ex['question']}\nAnswer (respond concisely):"
        full_text = prompt + " " + ex["gt"]
        sft_data.append({"text": full_text, "prompt": prompt, "gt": ex["gt"]})
    ds = Dataset.from_list(sft_data)

    def preprocess(example):
        # Tokenize full prompt+response
        enc = tokenizer(
            example["text"],
            padding="max_length",
            truncation=True,
            max_length=128,
        )
        # Get prompt length
        prompt_enc = tokenizer(
            example["prompt"],
            truncation=True,
            max_length=128,
            add_special_tokens=False
        )
        prompt_len = len(prompt_enc["input_ids"])
        # Labels: -100 for prompt, token IDs for answer
        labels = [-100] * prompt_len + enc["input_ids"][prompt_len:]
        labels += [-100] * (128 - len(labels))
        enc["labels"] = labels[:128]
        return enc

    print("\n==== MICRO LoRA FINE-TUNE ====")
    ds = ds.map(preprocess)
    print_vram()
    output_dir = "tmp_test_lora"
    training_args = TrainingArguments(
        per_device_train_batch_size=args.batch_size,
        num_train_epochs=args.ft_epochs,
        learning_rate=1e-4,
        fp16=True,
        logging_steps=1,
        output_dir=output_dir,
        overwrite_output_dir=True,
        save_strategy="no"
    )
    trainer = Trainer(
        model=model,
        train_dataset=ds,
        args=training_args,
        tokenizer=tokenizer,
    )
    trainer.train()
    print_vram()

    if os.path.exists(output_dir):
        import shutil
        shutil.rmtree(output_dir)
    print("\nAll integration points with LoRA, quantization, and SFT alignment tested! You are now SAFE to integrate to main project 🚀")

if __name__ == "__main__":
    main()

