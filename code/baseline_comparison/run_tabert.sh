#!/bin/sh

seed=$1
variant="baseline_tabert_${seed}"
model_save_file="../../models/model_${variant}.bin"
res_file="res_${variant}.pkl"
out_file="out_${variant}"
err_file="err_${variant}"

python -u train_baselines.py --seed $seed --model tabert --num_epochs 15 --batch_size 2 --lm_lr 3e-6 --lr 1e-5 --e_loss_lambda 1.0 --model_save_file $model_save_file --res_file $res_file >> $out_file 2>> $err_file
