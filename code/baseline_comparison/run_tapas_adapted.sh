#!/bin/sh

seed=$1
variant="baseline_tapas_adapted_${seed}"
model_save_file="../../models/model_${variant}.bin"
res_file="res_${variant}.pkl"
out_file="out_${variant}"
err_file="err_${variant}"

python -u train_baselines.py --seed $seed --model tapas_adapted --num_epochs 15 --lm_lr 2e-5 --lr 1e-3 --e_loss_lambda 1.0 --model_save_file $model_save_file --res_file $res_file >> $out_file 2>> $err_file
