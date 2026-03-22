import argparse
import importlib
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

CONTEXT_TOKEN = "\uEE02"
INPUT_START_TOKEN = "\uEE00"
OUTPUT_START_TOKEN = "\uEE01"


def build_prompt(left_context: str, input_kana: str) -> str:
    return f"{CONTEXT_TOKEN}{left_context}{INPUT_START_TOKEN}{input_kana}{OUTPUT_START_TOKEN}"


def resolve_gguf_path(model_path: str) -> Optional[str]:
    p = Path(model_path).expanduser()
    if p.is_file() and p.suffix.lower() == ".gguf":
        return str(p)

    if not p.is_dir():
        return None

    candidates = sorted(p.glob("*.gguf"))
    if not candidates:
        return None

    preferred_names = ["model-Q5_K_M.gguf", "model-Q4_K_M.gguf", "model-f16.gguf"]
    for name in preferred_names:
        for cand in candidates:
            if cand.name == name:
                return str(cand)

    return str(candidates[0])


def convert_hf(model_id: str, input_kana: str, left_context: str = "", max_new_tokens: int = 64) -> str:
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    model.eval()

    prompt = build_prompt(left_context, input_kana)
    input_ids = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")

    if torch.cuda.is_available():
        model = model.cuda()
        input_ids = input_ids.cuda()
    elif torch.backends.mps.is_available():
        model = model.to("mps")
        input_ids = input_ids.to("mps")

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

    generated = output_ids[0][input_ids.shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True)


def convert_gguf(
    model_gguf: str,
    input_kana: str,
    left_context: str = "",
    max_new_tokens: int = 64,
    n_ctx: int = 512,
    threads: int = 4,
    n_gpu_layers: int = 0,
) -> str:
    try:
        llama_cpp = importlib.import_module("llama_cpp")
        Llama = getattr(llama_cpp, "Llama")
    except Exception as e:
        raise SystemExit(
            "GGUF inference requires llama-cpp-python. Install with: pip install llama-cpp-python"
        ) from e

    llm = Llama(
        model_path=model_gguf,
        n_ctx=n_ctx,
        n_threads=max(1, threads),
        n_gpu_layers=n_gpu_layers,
        verbose=False,
    )

    prompt = build_prompt(left_context, input_kana)
    out = llm(
        prompt,
        max_tokens=max_new_tokens,
        temperature=0.0,
        top_k=1,
        top_p=1.0,
        repeat_penalty=1.0,
        stop=[CONTEXT_TOKEN, INPUT_START_TOKEN, OUTPUT_START_TOKEN],
        echo=False,
    )
    return out["choices"][0]["text"].strip()


def convert(
    model_path: str,
    input_kana: str,
    left_context: str = "",
    max_new_tokens: int = 64,
    n_ctx: int = 512,
    threads: int = 4,
    n_gpu_layers: int = 0,
) -> str:
    gguf_path = resolve_gguf_path(model_path)
    if gguf_path:
        return convert_gguf(
            gguf_path,
            input_kana,
            left_context=left_context,
            max_new_tokens=max_new_tokens,
            n_ctx=n_ctx,
            threads=threads,
            n_gpu_layers=n_gpu_layers,
        )

    return convert_hf(
        model_path,
        input_kana,
        left_context=left_context,
        max_new_tokens=max_new_tokens,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="HF repo/local dir, GGUF file, or GGUF dir")
    parser.add_argument("--input", required=True, help="katakana input")
    parser.add_argument("--left_context", default="")
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--n_ctx", type=int, default=512, help="context length for GGUF backend")
    parser.add_argument("--threads", type=int, default=4, help="CPU threads for GGUF backend")
    parser.add_argument("--n_gpu_layers", type=int, default=0, help="GPU layers for GGUF backend")
    args = parser.parse_args()

    text = convert(
        args.model,
        args.input,
        left_context=args.left_context,
        max_new_tokens=args.max_new_tokens,
        n_ctx=args.n_ctx,
        threads=args.threads,
        n_gpu_layers=args.n_gpu_layers,
    )
    print(text)


if __name__ == "__main__":
    main()
