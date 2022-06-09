#!/bin/sh

seed=$1
variant="gnn1_discomat_${seed}"
model_save_file="../../models/model_${variant}.bin"
res_file="res_${variant}.pkl"
out_file="out_${variant}"
err_file="err_${variant}"

python -u train_gnn_1.py --seed $seed --hidden_layer_sizes 256 128 64 --num_heads 4 4 4 --num_epochs 15 --use_regex_feat --use_max_freq_feat --lr 3e-4 --add_constraint --c_loss_lambda 50.0 --gid_loss_lambda 1.0 --model_save_file $model_save_file --res_file $res_file >> $out_file 2>> $err_file
