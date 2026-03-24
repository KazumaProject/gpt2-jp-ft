# Result

### 200000

#### command:

```bash
python train_pua_compatible.py --max_train_samples 200000 --num_train_epochs 1 --hardware gtx1060 --save_steps 100000000 --save_total_limit 1 --learning_rate 2e-5

python eval_ajimee_gguf.py   --model_gguf ./outputs/gpt2-kanakanji-pua-gguf/model-Q4_K_M.gguf   --limit 200   --save_predictions ./predictions.gguf.jsonl   --result_json ./result.gguf.json
```

#### Output:

```json

"with_context": {
    "n": 100,
    "accuracy_at1": 0.14,
    "avg_min_cer": 0.3680904970469264
  },
  "without_context": {
    "n": 100,
    "accuracy_at1": 0.15,
    "avg_min_cer": 0.3638288256273299
  }

```
### 500000

```bash

python train_pua_compatible.py --max_train_samples 500000 --num_train_epochs 1 --hardware gtx1060 --save_steps 100000000 --save_total_limit 1 --learning_rate 2e-5 --output_dir ./outputs/gpt2-kanakanji-pua-500000

```

```bash

=== AJIMEE-Bench result (greedy) ===
overall: n=200, accuracy_at1=0.2550, avg_min_cer=0.2164
with_context: n=100, accuracy_at1=0.2400, avg_min_cer=0.2169
without_context: n=100, accuracy_at1=0.2700, avg_min_cer=0.2160

```

### 1000000

```bash
=== AJIMEE-Bench result (gguf, greedy) ===
overall: n=200, accuracy_at1=0.3500, avg_min_cer=0.1413
with_context: n=100, accuracy_at1=0.3600, avg_min_cer=0.1456
without_context: n=100, accuracy_at1=0.3400, avg_min_cer=0.1371
```