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
