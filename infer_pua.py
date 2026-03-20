import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

CONTEXT_TOKEN = "\uEE02"
INPUT_START_TOKEN = "\uEE00"
OUTPUT_START_TOKEN = "\uEE01"


def convert(model_id: str, input_kana: str, left_context: str = "", max_new_tokens: int = 64) -> str:
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    model.eval()

    prompt = f"{CONTEXT_TOKEN}{left_context}{INPUT_START_TOKEN}{input_kana}{OUTPUT_START_TOKEN}"
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="HF repo or local model dir")
    parser.add_argument("--input", required=True, help="katakana input")
    parser.add_argument("--left_context", default="")
    parser.add_argument("--max_new_tokens", type=int, default=64)
    args = parser.parse_args()

    text = convert(args.model, args.input, args.left_context, args.max_new_tokens)
    print(text)


if __name__ == "__main__":
    main()
