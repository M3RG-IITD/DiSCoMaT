#!/bin/sh

seed=$1
variant="gnn1_discomat_wo_constraints_${seed}"
model_save_file="../../models/model_${variant}.bin"
res_file="res_${variant}.pkl"
out_file="out_${variant}"
err_file="err_${variant}"

python -u train_gnn_1.py --seed $seed --hidden_layer_sizes 128 64 64 --num_heads 4 4 2 --num_epochs 15 --lr 3e-4 --gid_loss_lambda 1.0 --use_regex_feat --use_max_freq_feat --model_save_file $model_save_file --res_file $res_file >> $out_file 2>> $err_file
