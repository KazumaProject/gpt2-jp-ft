import argparse
import json
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

CONTEXT_TOKEN = "\uEE02"
INPUT_START_TOKEN = "\uEE00"
OUTPUT_START_TOKEN = "\uEE01"


@dataclass
class Example:
    input_kana: str
    output: str
    left_context: str


def build_prompt(left_context: str, input_kana: str) -> str:
    return f"{CONTEXT_TOKEN}{left_context}{INPUT_START_TOKEN}{input_kana}{OUTPUT_START_TOKEN}"


def normalize_text(text: str) -> str:
    return text.strip()


def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            cur.append(min(cur[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost))
        prev = cur
    return prev[-1]


def cer(pred: str, ref: str) -> float:
    r = normalize_text(ref)
    p = normalize_text(pred)
    if len(r) == 0:
        return 0.0 if len(p) == 0 else 1.0
    return levenshtein(p, r) / len(r)


def iter_examples_from_hf(dataset: str, split: str, limit: Optional[int]) -> Iterable[Example]:
    ds = load_dataset(dataset, split=split)
    if limit is not None:
        ds = ds.select(range(min(limit, len(ds))))
    for row in ds:
        left_context = row.get("left_context") or ""
        yield Example(input_kana=row["input"], output=row["output"], left_context=left_context)


def iter_examples_from_jsonl(path: str, limit: Optional[int]) -> Iterable[Example]:
    count = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            left_context = row.get("left_context") or ""
            yield Example(input_kana=row["input"], output=row["output"], left_context=left_context)
            count += 1
            if limit is not None and count >= limit:
                break


def load_examples(args: argparse.Namespace) -> Iterable[Example]:
    if args.eval_jsonl:
        return iter_examples_from_jsonl(args.eval_jsonl, args.limit)
    return iter_examples_from_hf(args.dataset, args.split, args.limit)


def main() -> None:
    parser = argparse.ArgumentParser(description="AJIMEE-like evaluation for PUA kana-kanji models")
    parser.add_argument("--model", required=True, help="HF repo or local model dir")
    parser.add_argument("--dataset", default="Miwa-Keita/zenz-v2.5-dataset")
    parser.add_argument("--split", default="train")
    parser.add_argument("--eval_jsonl", default=None, help="Optional local JSONL with input/output/left_context")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--save_predictions", default=None, help="Optional JSONL path to save prediction details")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model)
    model.eval()

    if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()):
        device = "cuda"
    elif args.device == "mps" or (args.device == "auto" and torch.backends.mps.is_available()):
        device = "mps"
    else:
        device = "cpu"

    model = model.to(device)

    n = 0
    exact = 0
    total_cer = 0.0

    n_ctx = 0
    exact_ctx = 0
    cer_ctx = 0.0

    n_noctx = 0
    exact_noctx = 0
    cer_noctx = 0.0

    pred_file = None
    if args.save_predictions:
        pred_file = open(args.save_predictions, "w", encoding="utf-8")

    try:
        for ex in load_examples(args):
            prompt = build_prompt(ex.left_context, ex.input_kana)
            input_ids = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt").to(device)

            with torch.no_grad():
                output_ids = model.generate(
                    input_ids,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                )

            generated = output_ids[0][input_ids.shape[1]:]
            pred = tokenizer.decode(generated, skip_special_tokens=True).strip()
            ref = ex.output.strip()

            c = cer(pred, ref)
            ok = int(pred == ref)

            n += 1
            exact += ok
            total_cer += c

            if ex.left_context:
                n_ctx += 1
                exact_ctx += ok
                cer_ctx += c
            else:
                n_noctx += 1
                exact_noctx += ok
                cer_noctx += c

            if pred_file is not None:
                record: Dict[str, object] = {
                    "input": ex.input_kana,
                    "left_context": ex.left_context,
                    "reference": ref,
                    "prediction": pred,
                    "exact": bool(ok),
                    "cer": c,
                }
                pred_file.write(json.dumps(record, ensure_ascii=False) + "\n")

            if n % 50 == 0:
                print(f"[progress] {n} examples")
    finally:
        if pred_file is not None:
            pred_file.close()

    if n == 0:
        print("[error] no evaluation examples found")
        return

    print("\n=== AJIMEE-like result (greedy) ===")
    print(f"overall: n={n}, exact_match={exact / n:.4f}, avg_cer={total_cer / n:.4f}")

    if n_ctx > 0:
        print(f"with_context: n={n_ctx}, exact_match={exact_ctx / n_ctx:.4f}, avg_cer={cer_ctx / n_ctx:.4f}")
    if n_noctx > 0:
        print(f"without_context: n={n_noctx}, exact_match={exact_noctx / n_noctx:.4f}, avg_cer={cer_noctx / n_noctx:.4f}")


if __name__ == "__main__":
    main()
