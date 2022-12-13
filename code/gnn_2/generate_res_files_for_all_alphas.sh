#!/bin/sh

# This script will use a trained model and generate results using various MID alpha thresholds (0.4, 0.5, 0.6, 0.7, 0.8)
# If the trained model file's name is model_gnn2_discomat_0.bin, then the model variant is gnn2_discomat and 0 is the seed
# Pass the same hidden_layer_sizes, num_heads, use_max_freq_feat, max_freq_emb_size and add_constraint flags as were used in training the model


model_variant="gnn2_discomat"
out_file="out_gen_all_res_${model_variant}"
err_file="err_gen_all_res_${model_variant}"

python -u generate_res_files_for_all_alphas.py --seeds 0 1 2 --hidden_layer_sizes 128 128 64 --num_heads 6 4 4 --use_max_freq_feat --max_freq_emb_size 128 --add_constraint --model_variant $model_variant >> $out_file 2>> $err_file


model_variant="gnn2_discomat_wo_features"
out_file="out_gen_all_res_${model_variant}"
err_file="err_gen_all_res_${model_variant}"

python -u generate_res_files_for_all_alphas.py --seeds 0 1 2 --hidden_layer_sizes 128 128 64 --num_heads 6 4 4 --add_constraint --use_caption --model_variant $model_variant >> $out_file 2>> $err_file


model_variant="gnn2_discomat_wo_constraints"
out_file="out_gen_all_res_${model_variant}"
err_file="err_gen_all_res_${model_variant}"

python -u generate_res_files_for_all_alphas.py --seeds 0 1 2 --hidden_layer_sizes 128 128 64 --num_heads 6 4 4 --use_max_freq_feat --max_freq_emb_size 128 --use_caption --model_variant $model_variant >> $out_file 2>> $err_file


model_variant="gnn2_discomat_wo_caption"
out_file="out_gen_all_res_${model_variant}"
err_file="err_gen_all_res_${model_variant}"

python -u generate_res_files_for_all_alphas.py --seeds 0 1 2 --hidden_layer_sizes 128 128 64 --num_heads 6 4 4 --add_constraint --use_max_freq_feat --max_freq_emb_size 128 --model_variant $model_variant >> $out_file 2>> $err_file


model_variant="gnn2_vdiscomat"
out_file="out_gen_all_res_${model_variant}"
err_file="err_gen_all_res_${model_variant}"

python -u generate_res_files_for_all_alphas.py --seeds 0 1 2 --hidden_layer_sizes 128 64 64 --num_heads 4 4 2 --model_variant $model_variant >> $out_file 2>> $err_file
