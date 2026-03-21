import argparse
import json
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

CONTEXT_TOKEN = "\uEE02"
INPUT_START_TOKEN = "\uEE00"
OUTPUT_START_TOKEN = "\uEE01"


@dataclass
class Example:
    input_kana: str
    expected_outputs: List[str]
    left_context: str
    index: str


def build_prompt(left_context: str, input_kana: str) -> str:
    return f"{CONTEXT_TOKEN}{left_context}{INPUT_START_TOKEN}{input_kana}{OUTPUT_START_TOKEN}"


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


def calculate_CER(reference: str, hypothesis: str) -> float:
    """Same CER definition as AJIMEE-Bench utils.py."""
    if len(reference) == 0:
        return float("inf")
    return levenshtein(reference, hypothesis) / len(reference)


def calculate_MinCER(references: Sequence[str], hypothesis: str) -> float:
    if len(references) == 0:
        return float("inf")
    return min(calculate_CER(reference, hypothesis) for reference in references)


def calculate_accuracy_at1(references: Sequence[str], attempt: str) -> int:
    return 1 if attempt in references else 0


def iter_examples_from_ajimee_json(path: str, limit: Optional[int]) -> Iterable[Example]:
    with open(path, "r", encoding="utf-8") as f:
        rows = json.load(f)

    if limit is not None:
        rows = rows[:limit]

    for row in rows:
        expected_outputs = [x.strip() for x in row.get("expected_output", []) if isinstance(x, str) and x.strip()]
        if len(expected_outputs) == 0:
            continue
        yield Example(
            input_kana=row["input"],
            expected_outputs=expected_outputs,
            left_context=(row.get("context_text") or ""),
            index=str(row.get("index", "")),
        )


def iter_examples_from_jsonl(path: str, limit: Optional[int]) -> Iterable[Example]:
    count = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)

            expected_outputs_raw = row.get("expected_output")
            if isinstance(expected_outputs_raw, list):
                expected_outputs = [x.strip() for x in expected_outputs_raw if isinstance(x, str) and x.strip()]
            elif isinstance(row.get("output"), str):
                expected_outputs = [row["output"].strip()]
            else:
                expected_outputs = []

            if len(expected_outputs) == 0:
                continue

            left_context = row.get("left_context") or row.get("context_text") or ""
            yield Example(
                input_kana=row["input"],
                expected_outputs=expected_outputs,
                left_context=left_context,
                index=str(row.get("index", "")),
            )
            count += 1
            if limit is not None and count >= limit:
                break


def ensure_ajimee_json(path: str) -> str:
    p = Path(path).expanduser()
    if p.exists():
        return str(p)

    p.parent.mkdir(parents=True, exist_ok=True)
    url = "https://raw.githubusercontent.com/azooKey/AJIMEE-Bench/main/JWTD_v2/v1/evaluation_items.json"
    print(f"[info] AJIMEE-Bench dataset not found at {p}; downloading from {url}")
    urllib.request.urlretrieve(url, str(p))
    return str(p)


def load_examples(args: argparse.Namespace) -> Iterable[Example]:
    if args.eval_jsonl:
        return iter_examples_from_jsonl(args.eval_jsonl, args.limit)
    ajimee_json = ensure_ajimee_json(args.ajimee_json)
    return iter_examples_from_ajimee_json(ajimee_json, args.limit)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run AJIMEE-Bench evaluation for PUA kana-kanji models")
    parser.add_argument("--model", required=True, help="HF repo or local model dir")
    parser.add_argument(
        "--ajimee_json",
        default="~/.cache/ajimee-bench/evaluation_items.json",
        help="Path to AJIMEE-Bench evaluation_items.json (auto-downloaded if missing)",
    )
    parser.add_argument(
        "--eval_jsonl",
        default=None,
        help="Optional custom JSONL (supports expected_output[] or output)",
    )
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
    acc_at1_sum = 0
    mincer_sum = 0.0

    n_ctx = 0
    acc_ctx = 0
    mincer_ctx = 0.0

    n_noctx = 0
    acc_noctx = 0
    mincer_noctx = 0.0

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
            references = [r.strip() for r in ex.expected_outputs]

            c = calculate_MinCER(references, pred)
            ok = calculate_accuracy_at1(references, pred)

            n += 1
            acc_at1_sum += ok
            mincer_sum += c

            if ex.left_context:
                n_ctx += 1
                acc_ctx += ok
                mincer_ctx += c
            else:
                n_noctx += 1
                acc_noctx += ok
                mincer_noctx += c

            if pred_file is not None:
                record: Dict[str, object] = {
                    "index": ex.index,
                    "input": ex.input_kana,
                    "left_context": ex.left_context,
                    "expected_output": references,
                    "prediction": pred,
                    "accuracy_at1": bool(ok),
                    "min_cer": c,
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

    print("\n=== AJIMEE-Bench result (greedy) ===")
    print(f"overall: n={n}, accuracy_at1={acc_at1_sum / n:.4f}, avg_min_cer={mincer_sum / n:.4f}")

    if n_ctx > 0:
        print(f"with_context: n={n_ctx}, accuracy_at1={acc_ctx / n_ctx:.4f}, avg_min_cer={mincer_ctx / n_ctx:.4f}")
    if n_noctx > 0:
        print(f"without_context: n={n_noctx}, accuracy_at1={acc_noctx / n_noctx:.4f}, avg_min_cer={mincer_noctx / n_noctx:.4f}")


if __name__ == "__main__":
    main()
