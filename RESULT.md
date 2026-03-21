# Result

## 200000

### command:
```bash
python train_pua_compatible.py --max_train_samples 200000 --num_train_epochs 1 --hardware gtx1060 --save_steps 100000000 --save_total_limit 1 --learning_rate 2e-5
```

### Output:
```bash

=== AJIMEE-Bench result (greedy) ===
overall: n=200, accuracy_at1=0.1650, avg_min_cer=0.3328
with_context: n=100, accuracy_at1=0.1700, avg_min_cer=0.3316
without_context: n=100, accuracy_at1=0.1600, avg_min_cer=0.3340

```