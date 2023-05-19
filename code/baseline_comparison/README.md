## DiSCoMaT baselines

Here, we train and evaluate on MCC-CI and NC tables. We run our experiments of three seeds: 0, 1, 2.

### Training

#### TaPas Baseline
```bash
for seed in 0 1 2; do
    bash run_tapas.sh $seed
done
```

#### TaPas-Adapted Baseline
```bash
for seed in 0 1 2; do
    bash run_tapas_adapted.sh $seed
done
```

#### TaBERT Baseline
```bash
for seed in 0 1 2; do
    bash run_tabert.sh $seed
done
```

#### TaBERT-Adapted Baseline
```bash
for seed in 0 1 2; do
    bash run_tabert_adapted.sh $seed
done
```

#### v-DiSCoMaT
```bash
for seed in 0 1 2; do
    bash run_gat.sh $seed
done
```

These five models can be run on all three seeds using `bash run.sh`.
All these experiments can be parallelized easily. Hyper-parameters can be modified from within .sh scripts.


### Evaluation
The following commands output the results averaged over three seeds for all models

```bash
python compute_results.py --model tapas
python compute_results.py --model tapas_adapted
python compute_results.py --model tabert
python compute_results.py --model tabert_adapted
python compute_results.py --model gat
```

### Simple Rule Based Model
This can be run by the the command:
```bash
python simple_rule_based_baseline.py
```
