#!/bin/sh

seed=$1
variant="gnn2_vdiscomat_${seed}"
model_save_file="../../models/model_${variant}.bin"
res_file="res_${variant}.pkl"
out_file="out_${variant}"
err_file="err_${variant}"

python -u train_gnn_2.py --seed $seed --hidden_layer_sizes 128 64 64 --num_heads 4 4 2 --num_epochs 15 --e_loss_lambda 1.0 --lr 1e-3 --lm_lr 2e-5 --model_save_file $model_save_file --res_file $res_file >> $out_file 2>> $err_file
