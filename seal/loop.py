# FILE: seal/loop.py

from __future__ import annotations
import gc, json, os, pickle, random
from pathlib import Path
from typing import Any, Dict, List, Optional
import hydra, torch
import torch.distributed as dist
from datasets import load_dataset
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, GenerationConfig
from seal.evaluate import eval_knowledge
from seal.generate_edits import sample_edits
from seal.prompts import build_knowledge_prompt
from seal.rl_restem import restem_round, semantic_sim
from seal.sft_lora import fine_tune_lora, fine_tune_lora_ddp

CONF_DIR = str(Path(__file__).resolve().parent.parent / "conf")
CKPT_DIR = Path(os.getenv("CKPT_DIR", "ckpt"))
CKPT_DIR.mkdir(parents=True, exist_ok=True)
STATE_PKL = CKPT_DIR / "state.pkl"
EXAMPLES_PKL = CKPT_DIR / "train_examples.pkl"
MODEL_DIR = CKPT_DIR / "policy"
TOKEN_DIR = CKPT_DIR / "tokenizer"
BATCH = int(os.getenv("GEN_BATCH", "100"))

def atomic_write(obj, fname: Path) -> None:
    tmp = fname.with_suffix(fname.suffix + ".tmp")
    pickle.dump(obj, open(tmp, "wb"))
    os.replace(tmp, fname)

def clean_llama_answer(txt: str) -> str:
    lines = [l for l in txt.splitlines() if l.strip() and not l.strip().startswith("<|")]
    return (lines[0] if lines else txt).lstrip("> ").lstrip("<|assistant|>").strip().rstrip(".")

def _hf_to_internal(ex) -> Dict[str, Any]:
    return dict(id=ex.get("id"), question=ex["question"], ground_truth=ex["answers"]["text"][0], answer="", metric=0.0, semantic_sim=0.0)

def _f1(pred: str, gold: str) -> float:
    g, p = gold.lower().split(), pred.lower().split()
    inter = len(set(g) & set(p))
    if inter == 0:
        return 100.0 if gold.strip().lower() in pred.strip().lower() else 0.0
    prec, rec = inter / len(p), inter / len(g)
    return 2 * prec * rec / (prec + rec) * 100.0

def _get_policy_class(name: str):
    return AutoModelForCausalLM

def load_model(path: Path):
    torch.cuda.empty_cache()
    gc.collect()
    
    if cfg.model.load_4bit:
        bnb = BitsAndBytesConfig(
            load_in_4bit=True, 
            bnb_4bit_use_double_quant=True, 
            bnb_4bit_compute_dtype=torch.float16
        )
        model = PolicyClass.from_pretrained(
            str(path), 
            device_map="auto",  # Let transformers handle device mapping
            quantization_config=bnb
        )
        # Ensure model is properly on GPU
        model = model.cuda()
    else:
        model = PolicyClass.from_pretrained(str(path), device_map="auto", torch_dtype="auto")
    
    return model


def load_model_ddp(path: Path, cfg):
    """Load model with proper DDP distribution"""
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    
    torch.cuda.set_device(local_rank)
    torch.cuda.empty_cache()
    gc.collect()
    
    PolicyClass = _get_policy_class(cfg.model.name)
    
    # Only load on rank 0, then broadcast
    if local_rank == 0:
        if cfg.model.load_4bit:
            bnb = BitsAndBytesConfig(
                load_in_4bit=True, 
                bnb_4bit_use_double_quant=True, 
                bnb_4bit_compute_dtype=torch.float16
            )
            model = PolicyClass.from_pretrained(
                str(path), 
                device_map={"": 0},  # Load on GPU 0 first
                quantization_config=bnb
            )
        else:
            model = PolicyClass.from_pretrained(
                str(path), 
                device_map={"": 0}, 
                torch_dtype="auto"
            )
        
        # Then move to appropriate device
        model = model.to(f"cuda:{local_rank}")
    else:
        # Other ranks wait and load from broadcast
        if cfg.model.load_4bit:
            bnb = BitsAndBytesConfig(
                load_in_4bit=True, 
                bnb_4bit_use_double_quant=True, 
                bnb_4bit_compute_dtype=torch.float16
            )
            model = PolicyClass.from_pretrained(
                str(path), 
                device_map={"": local_rank}, 
                quantization_config=bnb
            )
        else:
            model = PolicyClass.from_pretrained(
                str(path), 
                device_map={"": local_rank}, 
                torch_dtype="auto"
            )
    
    # Synchronize all processes
    if dist.is_initialized():
        dist.barrier()
    
    return model

def safe_round(pol, tok, ctx_batch, cfg_dict, r_num):
    try:
        torch.cuda.empty_cache()
        
        # Ensure model is properly on device
        if hasattr(pol, 'cuda'):
            pol = pol.cuda()
        
        return restem_round(
            pol, tok, ctx_batch, cfg_dict,
            generate_edits_fn=sample_edits,
            evaluate_fn=eval_knowledge,
            fine_tune_fn=fine_tune_lora,
            round_num=r_num
        )
    except Exception as e:
        logger.error(f"Training error during round {r_num}: {str(e)} — skipping fine-tune")
        # Print full traceback for debugging
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        torch.cuda.empty_cache()
        gc.collect()
        return pol, None, []

@hydra.main(config_path=CONF_DIR, config_name="knowledge", version_base=None)
def main(cfg: DictConfig):
    # Initialize distributed training if running with torchrun
    if os.environ.get('LOCAL_RANK') is not None:
        if not dist.is_initialized():
            dist.init_process_group(backend='nccl')
        local_rank = int(os.environ['LOCAL_RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        torch.cuda.set_device(local_rank)
        logger.info(f"[Rank {local_rank}/{world_size}] Initialized DDP")
    else:
        local_rank = 0
        world_size = 1
        logger.info("Running in single-GPU mode")

    # Only log config on rank 0 to avoid spam
    if local_rank == 0:
        logger.info("Config:\n" + OmegaConf.to_yaml(cfg))
    
    PolicyClass = _get_policy_class(cfg.model.name)

    # Use DDP-aware model loading if in distributed mode
    if dist.is_initialized():
        def load_model(path: Path):
            return load_model_ddp(path, cfg)
    else:
        def load_model(path: Path):
            torch.cuda.empty_cache()
            gc.collect()
            if cfg.model.load_4bit:
                bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16)
                return PolicyClass.from_pretrained(str(path), device_map={"": torch.cuda.current_device()}, quantization_config=bnb)
            return PolicyClass.from_pretrained(str(path), device_map="auto", torch_dtype="auto")

    # Load model and tokenizer
    if STATE_PKL.exists():
        state = pickle.load(open(STATE_PKL, "rb"))
        train_examples = state["train_examples"]
        policy = load_model(MODEL_DIR)
        tokenizer = AutoTokenizer.from_pretrained(str(TOKEN_DIR), use_fast=True)
        if local_rank == 0:
            logger.success("✓ Resumed from state.pkl")
    elif EXAMPLES_PKL.exists():
        train_examples = pickle.load(open(EXAMPLES_PKL, "rb"))
        policy = load_model(Path(cfg.model.name))
        tokenizer = AutoTokenizer.from_pretrained(cfg.model.name, use_fast=True)
        state = None
        if local_rank == 0:
            logger.success("✓ train_examples restored, will evaluate for baseline.")
    else:
        policy = load_model(Path(cfg.model.name))
        tokenizer = AutoTokenizer.from_pretrained(cfg.model.name, use_fast=True)
        train_examples = None
        state = None

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # Load dataset (only on rank 0, then broadcast)
    if local_rank == 0:
        raw = load_dataset(cfg.dataset.name, split=cfg.dataset.train_slice)
        if train_examples is None:
            train_examples = [_hf_to_internal(ex) for ex in raw]
    
    # Broadcast train_examples to all ranks if needed
    if dist.is_initialized():
        if local_rank == 0:
            train_examples_list = [train_examples]
        else:
            train_examples_list = [None]
        dist.broadcast_object_list(train_examples_list, src=0)
        train_examples = train_examples_list[0]

    # Generate missing answers (only on rank 0)
    missing_idx = [i for i, ex in enumerate(train_examples) if not str(ex.get("answer", "")).strip()]
    if missing_idx and local_rank == 0:
        logger.info(f"Generating answers for {len(missing_idx)} missing examples…")
        for bs in tqdm(range(0, len(missing_idx), BATCH), desc="Generating Initial Answers"):
            gc.collect()
            torch.cuda.empty_cache()
            idx_slice = missing_idx[bs : bs + BATCH]
            prompts = [build_knowledge_prompt(train_examples[i]["question"]) for i in idx_slice]
            inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=128).to(policy.device)
            gen_cfg = GenerationConfig(do_sample=True, temperature=0.7, top_p=0.9, max_new_tokens=32, pad_token_id=tokenizer.pad_token_id)
            
            with torch.no_grad():
                out = policy.generate(**inputs, generation_config=gen_cfg)
            
            for j, seq in enumerate(out):
                ex = train_examples[idx_slice[j]]
                gen_txt = seq[inputs.input_ids.shape[-1]:]
                answer = clean_llama_answer(tokenizer.decode(gen_txt, skip_special_tokens=True))
                ex["answer"] = answer
                ex["metric"] = _f1(answer, ex["ground_truth"])
                ex["semantic_sim"] = semantic_sim(answer, ex["ground_truth"])
        
        atomic_write(train_examples, EXAMPLES_PKL)
    
    # Wait for all processes to finish initial answer generation
    if dist.is_initialized():
        dist.barrier()
        # Reload train_examples on non-rank-0 processes
        if local_rank != 0:
            train_examples = pickle.load(open(EXAMPLES_PKL, "rb"))

    # Initialize baseline (only on rank 0)
    if state is None and local_rank == 0:
        base_f1, *_ = eval_knowledge(policy, tokenizer, train_examples[:20])
        state = dict(train_examples=train_examples, current_metric=base_f1, next_round=1)
        atomic_write(state, STATE_PKL)
        policy.save_pretrained(str(MODEL_DIR))
        tokenizer.save_pretrained(str(TOKEN_DIR))
    
    # Broadcast state to all ranks
    if dist.is_initialized():
        if local_rank == 0:
            state_list = [state]
        else:
            state_list = [None]
        dist.broadcast_object_list(state_list, src=0)
        state = state_list[0]
        
        # Non-rank-0 processes reload policy
        if local_rank != 0:
            policy = load_model(MODEL_DIR)

    # Main training loop
    for r in range(state["next_round"], cfg.rl.max_rounds + 1):
        if local_rank == 0:
            logger.info(f"===== SEAL round {r}/{cfg.rl.max_rounds} =====")
        
        # Sample batch (same across all ranks)
        random.seed(r)  # Ensure same sampling across ranks
        batch = random.sample(train_examples, cfg.dataset.ctx_per_round)
        
        policy_new, metric_new, winners = safe_round(policy, tokenizer, batch, OmegaConf.to_container(cfg, resolve=True), r)
        
        # Only rank 0 handles state updates
        if local_rank == 0:
            prev_metric = state["current_metric"]
            if metric_new is None or metric_new <= prev_metric:
                logger.warning(f"✗ Round {r} no gain (prev={prev_metric:.3f}) → rollback")
                delta = 0.0
            else:
                policy = policy_new
                delta = metric_new - prev_metric
                state["current_metric"] = metric_new
                logger.success(f"✓ Round {r} improved by {delta:.3f}")

            for w in winners:
                for ex in train_examples:
                    if ex.get("id") == w.get("id") or ex["question"] == w["question"]:
                        ex.update(w)

            state["train_examples"] = train_examples
            atomic_write(state, STATE_PKL)
            if policy_new is not None:
                policy_new.save_pretrained(str(MODEL_DIR))
            state["next_round"] = r + 1
            atomic_write(state, STATE_PKL)
        
        # Synchronize all processes
        if dist.is_initialized():
            dist.barrier()
            # Non-rank-0 processes update policy if improved
            if local_rank != 0 and policy_new is not None:
                policy = load_model(MODEL_DIR)
        
        gc.collect()

    if local_rank == 0:
        logger.success("SEAL training loop completed.")

    # Cleanup distributed training
    if dist.is_initialized():
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
