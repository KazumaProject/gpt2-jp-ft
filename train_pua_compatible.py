import argparse
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

CONTEXT_TOKEN = "\uEE02"
INPUT_START_TOKEN = "\uEE00"
OUTPUT_START_TOKEN = "\uEE01"


@dataclass
class HardwareProfile:
    name: str
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    fp16: bool
    bf16: bool


@dataclass
class Preset:
    max_length: int
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    gradient_checkpointing: bool
    dataloader_num_workers: int


def detect_hardware_profile(force: Optional[str] = None) -> HardwareProfile:
    if force == "mps":
        return HardwareProfile("mps", per_device_train_batch_size=2, gradient_accumulation_steps=16, fp16=False, bf16=False)
    if force == "gtx1060":
        return HardwareProfile("gtx1060", per_device_train_batch_size=1, gradient_accumulation_steps=32, fp16=False, bf16=False)

    if torch.backends.mps.is_available():
        return HardwareProfile("mps", per_device_train_batch_size=2, gradient_accumulation_steps=16, fp16=False, bf16=False)

    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0).lower()
        if "1060" in device_name:
            return HardwareProfile("gtx1060", per_device_train_batch_size=1, gradient_accumulation_steps=32, fp16=False, bf16=False)

        major, _ = torch.cuda.get_device_capability(0)
        bf16 = major >= 8
        fp16 = not bf16
        return HardwareProfile("cuda", per_device_train_batch_size=2, gradient_accumulation_steps=16, fp16=fp16, bf16=bf16)

    return HardwareProfile("cpu", per_device_train_batch_size=1, gradient_accumulation_steps=64, fp16=False, bf16=False)


def build_prompt(left_context: Optional[str], input_kana: str) -> str:
    left = left_context if left_context else ""
    return f"{CONTEXT_TOKEN}{left}{INPUT_START_TOKEN}{input_kana}{OUTPUT_START_TOKEN}"


def resolve_preset(profile: HardwareProfile, preset: str) -> Preset:
    if preset == "fast":
        if profile.name == "gtx1060":
            return Preset(max_length=160, per_device_train_batch_size=1, gradient_accumulation_steps=16, gradient_checkpointing=True, dataloader_num_workers=1)
        if profile.name == "mps":
            return Preset(max_length=192, per_device_train_batch_size=2, gradient_accumulation_steps=8, gradient_checkpointing=False, dataloader_num_workers=2)
        return Preset(max_length=192, per_device_train_batch_size=2, gradient_accumulation_steps=8, gradient_checkpointing=False, dataloader_num_workers=2)

    if preset == "oom_safe":
        if profile.name == "gtx1060":
            return Preset(max_length=128, per_device_train_batch_size=1, gradient_accumulation_steps=48, gradient_checkpointing=True, dataloader_num_workers=1)
        if profile.name == "mps":
            return Preset(max_length=160, per_device_train_batch_size=1, gradient_accumulation_steps=32, gradient_checkpointing=True, dataloader_num_workers=1)
        return Preset(max_length=160, per_device_train_batch_size=1, gradient_accumulation_steps=24, gradient_checkpointing=True, dataloader_num_workers=1)

    # balanced
    if profile.name == "gtx1060":
        return Preset(max_length=192, per_device_train_batch_size=1, gradient_accumulation_steps=32, gradient_checkpointing=True, dataloader_num_workers=1)
    if profile.name == "mps":
        return Preset(max_length=256, per_device_train_batch_size=2, gradient_accumulation_steps=16, gradient_checkpointing=False, dataloader_num_workers=2)
    return Preset(max_length=256, per_device_train_batch_size=2, gradient_accumulation_steps=16, gradient_checkpointing=False, dataloader_num_workers=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune GPT-2 for kana-kanji conversion with gpt2-kanakanji-compatible PUA tokens")
    parser.add_argument("--base_model", default="ku-nlp/gpt2-small-japanese-char")
    parser.add_argument("--dataset", default="Miwa-Keita/zenz-v2.5-dataset")
    parser.add_argument("--dataset_split", default="train")
    parser.add_argument("--output_dir", default="./outputs/gpt2-kanakanji-pua")
    parser.add_argument("--max_length", type=int, default=None)
    parser.add_argument("--num_train_epochs", type=float, default=1.0)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--warmup_ratio", type=float, default=0.01)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--logging_steps", type=int, default=50)
    parser.add_argument("--save_steps", type=int, default=2000)
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--max_train_samples", type=int, default=None, help="Use small subset for smoke test")
    parser.add_argument("--num_proc", type=int, default=max(1, (os.cpu_count() or 4) // 2))
    parser.add_argument("--hardware", choices=["auto", "mps", "gtx1060"], default="auto")
    parser.add_argument("--preset", choices=["balanced", "oom_safe", "fast"], default="balanced")
    parser.add_argument("--per_device_train_batch_size", type=int, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=None)
    parser.add_argument("--dataloader_num_workers", type=int, default=None)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--hub_model_id", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    forced = None if args.hardware == "auto" else args.hardware
    profile = detect_hardware_profile(forced)
    preset = resolve_preset(profile, args.preset)

    print(f"[info] hardware profile: {profile.name}")

    max_length = args.max_length if args.max_length is not None else preset.max_length
    batch_size = args.per_device_train_batch_size if args.per_device_train_batch_size is not None else preset.per_device_train_batch_size
    grad_accum = args.gradient_accumulation_steps if args.gradient_accumulation_steps is not None else preset.gradient_accumulation_steps
    num_workers = args.dataloader_num_workers if args.dataloader_num_workers is not None else preset.dataloader_num_workers
    use_gc = args.gradient_checkpointing or preset.gradient_checkpointing

    print(
        "[info] train settings: "
        f"preset={args.preset}, max_length={max_length}, "
        f"batch={batch_size}, grad_accum={grad_accum}, "
        f"grad_checkpointing={use_gc}"
    )

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)

    # Keep the exact token chars for gpt2-kanakanji compatibility.
    tokenizer.add_special_tokens({"additional_special_tokens": [CONTEXT_TOKEN, INPUT_START_TOKEN, OUTPUT_START_TOKEN]})

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.bos_token

    model = AutoModelForCausalLM.from_pretrained(args.base_model)
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.eos_token_id = tokenizer.eos_token_id

    if use_gc:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    ds = load_dataset(args.dataset, split=args.dataset_split)
    if args.max_train_samples is not None:
        ds = ds.select(range(min(args.max_train_samples, len(ds))))

    def preprocess(ex: Dict[str, str]) -> Dict[str, List[int]]:
        prompt = build_prompt(ex.get("left_context"), ex["input"])
        target = ex["output"] + tokenizer.eos_token

        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        target_ids = tokenizer.encode(target, add_special_tokens=False)

        input_ids = (prompt_ids + target_ids)[:max_length]
        labels = ([-100] * len(prompt_ids) + target_ids)[:max_length]
        attention_mask = [1] * len(input_ids)

        pad_len = max_length - len(input_ids)
        if pad_len > 0:
            input_ids.extend([tokenizer.pad_token_id] * pad_len)
            labels.extend([-100] * pad_len)
            attention_mask.extend([0] * pad_len)

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }

    cols = ds.column_names
    tokenized = ds.map(preprocess, remove_columns=cols, num_proc=args.num_proc, desc="Tokenizing")

    train_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        do_train=True,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        fp16=profile.fp16,
        bf16=profile.bf16,
        dataloader_num_workers=num_workers,
        report_to="none",
        remove_unused_columns=False,
        optim="adamw_torch",
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id,
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=tokenized,
        tokenizer=tokenizer,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    metrics = trainer.state.log_history
    if metrics:
        last = metrics[-1]
        if "loss" in last:
            ppl = math.exp(last["loss"]) if last["loss"] < 20 else float("inf")
            print(f"[info] last_loss={last['loss']:.4f}, approx_ppl={ppl:.2f}")

    print(f"[done] model saved to: {args.output_dir}")
    print("[done] compatible prompt format:")
    print("  prompt = CONTEXT + left_context + INPUT_START + input_kana + OUTPUT_START")


if __name__ == "__main__":
    main()
