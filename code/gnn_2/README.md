## Training GNN2

### Training DiSCoMaT GNN2
```bash
for seed in 0 1 2; do
    bash run_discomat.sh $seed
done
```

### Training DiSCoMaT w/o features GNN2
```bash
for seed in 0 1 2; do
    bash run_discomat_wo_features.sh $seed
done
```

### Training DiSCoMaT w/o constraints GNN2
```bash
for seed in 0 1 2; do
    bash run_discomat_wo_constraints.sh $seed
done
```

### Training DiSCoMaT w/o caption GNN2
```bash
for seed in 0 1 2; do
    bash run_discomat_wo_caption.sh $seed
done
```

### Training v-DiSCoMaT GNN2
```bash
for seed in 0 1 2; do
    bash run_vdiscomat.sh $seed
done
```

All the above experiments can be parallelized easily. Hyper-parameters can be modified from within .sh scripts.
