#!/bin/sh

seed=$1
variant="gnn2_discomat_${seed}"
model_save_file="../../models/model_${variant}.bin"
res_file="res_${variant}.pkl"
out_file="out_${variant}"
err_file="err_${variant}"

python -u train_gnn_2.py --seed $seed --hidden_layer_sizes 128 128 64 --num_heads 6 4 4 --num_epochs 20 --e_loss_lambda 1.0 --lr 3e-4 --lm_lr 2e-5 --add_constraint --c_loss_lambda 30.0 --use_max_freq_feat --max_freq_emb_size 128 --use_caption --model_save_file $model_save_file --res_file $res_file >> $out_file 2>> $err_file
