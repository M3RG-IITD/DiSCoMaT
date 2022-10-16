#!/bin/sh

for model in gat tapas tapas_adapted tabert tabert_adapted; do
    for seed in 0 1 2; do
        echo $model $seed
        bash "run_${model}.sh" $seed
    done
done
