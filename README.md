# gpt2-kanakanji compatible PUA training

This setup reproduces a kana-kanji fine-tuning pipeline with the same PUA prompt tokens used by `yuuki14202028/gpt2-kanakanji`.

Hardware targets:

- Apple Silicon (M1/M2, MPS)
- GTX1060 (low-VRAM safe defaults)

## Prompt format

- CONTEXT token: U+EE02
- INPUT_START token: U+EE00
- OUTPUT_START token: U+EE01

Prompt:

`{CONTEXT}<left_context>{INPUT_START}<input>{OUTPUT_START}`

Target:

`<output></s>`

Training loss is applied only on target tokens.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install --index-url https://download.pytorch.org/whl/cu118 torch==2.1.2
pip install -r requirements.txt
```

For GGUF evaluation pipeline, install `llama-cpp-python` too:

```bash
pip install llama-cpp-python
```

For GTX1060 (sm_61), avoid `transformers` 5.x with the latest `torch` CUDA 12.8 wheels,
because that combination drops this GPU architecture.

## Smoke test (recommended first)

```bash
python train_pua_compatible.py \
  --max_train_samples 50000 \
  --num_train_epochs 0.2 \
  --preset fast
```

## Full-ish run on Apple Silicon (M1/M2)

```bash
python train_pua_compatible.py \
  --hardware mps \
  --preset balanced \
  --num_train_epochs 1.0 \
  --learning_rate 2e-5
```

## Full-ish run on GTX1060

```bash
python train_pua_compatible.py \
  --hardware gtx1060 \
  --preset oom_safe \
  --num_train_epochs 1.0 \
  --learning_rate 2e-5
```

## Presets

- balanced: quality/speed balance (default)
- oom_safe: minimizes OOM risk
- fast: shorter sequence and lighter accumulation for quick iteration

You can always override internals manually:

```bash
python train_pua_compatible.py \
  --hardware gtx1060 \
  --preset oom_safe \
  --max_length 160 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 24
```

## Time reduction tips

```bash
# Quick signal in short time
python train_pua_compatible.py --preset fast --max_train_samples 200000 --num_train_epochs 0.4

# Reduce tokenizer preprocessing worker overhead on low-memory hosts
python train_pua_compatible.py --preset oom_safe --num_proc 2 --dataloader_num_workers 1
```

## Inference

```bash
python infer_pua.py --model ./outputs/gpt2-kanakanji-pua --input ニホンゴ
python infer_pua.py --model ./outputs/gpt2-kanakanji-pua --input ノイライガクルヨウニ --left_context きっかけで、漫画の仕事
```

## AJIMEE-Bench evaluation (greedy)

```bash
# Evaluate official AJIMEE-Bench (auto-download if missing)
python eval_ajimee_like.py --model ./outputs/gpt2-kanakanji-pua --limit 200

# Evaluate local benchmark JSONL (AJIMEE-style schema)
python eval_ajimee_like.py --model ./outputs/gpt2-kanakanji-pua --eval_jsonl ./bench.jsonl --limit 200 --save_predictions ./predictions.jsonl
```

## Train -> GGUF -> Quantize -> AJIMEE-Bench (one command)

Prerequisite:

- `llama.cpp` is cloned and built (`llama-quantize` available)
- `llama-cpp-python` installed for GGUF evaluation

Example:

```bash
python train_pua_compatible.py --max_train_samples 200000 --num_train_epochs 1 --hardware gtx1060 --save_steps 100000000 --save_total_limit 1 --learning_rate 2e-5
```

Artifacts are saved under:

- GGUF / quantized files: `<output_dir>-gguf/`
- AJIMEE predictions/results: `<output_dir>-gguf/ajimee/`

## GGUF export/quantize only (separate script)

Prepare `llama.cpp` (ensan-hcl fork) under parent directory:

```bash
git clone --depth 1 https://github.com/ensan-hcl/llama.cpp ../llama.cpp

# Option A: build with make (no cmake required)
make -C ../llama.cpp -j

# Option B: build with cmake
cmake -S ../llama.cpp -B ../llama.cpp/build
cmake --build ../llama.cpp/build -j
```

```bash
python export_quantize_gguf.py \
  --model_dir ./outputs/gpt2-kanakanji-pua \
  --llama_cpp_dir ../llama.cpp \
  --gguf_outtype f16 \
  --quantize_types Q4_K_M,Q5_K_M
```

You can also run GGUF evaluation directly:

```bash
python eval_ajimee_gguf.py \
  --model_gguf ./outputs/gpt2-kanakanji-pua-gguf/model-Q4_K_M.gguf \
  --limit 200 \
  --save_predictions ./predictions.gguf.jsonl \
  --result_json ./result.gguf.json
```

Expected JSONL schema:

- input: katakana input
- expected_output: list of acceptable conversion candidates (preferred)
- output: single reference conversion (backward-compatible)
- left_context or context_text: optional left context (empty or null allowed)

Metrics:

- accuracy_at1
- avg_min_cer (minimum character error rate over expected_output)
- with_context / without_context breakdown

## Notes

- This is a practical reproduction recipe. Exact match with `zenz-v2.5-small` is difficult because official training hyperparameters are not fully public.
- License inheritance and dataset terms must be followed when redistributing.
