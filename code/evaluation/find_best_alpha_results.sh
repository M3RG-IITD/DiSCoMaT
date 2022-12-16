#!/bin/sh

# DiSCoMat
python find_best_alpha_results.py --gnn1_variant gnn1_discomat --gnn2_variant gnn2_discomat --alphas 0.4 0.5 0.6 0.7 0.8

# DiSCoMat w/o features
python find_best_alpha_results.py --gnn1_variant gnn1_discomat_wo_features --gnn2_variant gnn2_discomat_wo_features --alphas 0.4 0.5 0.6 0.7 0.8

# DiSCoMat w/o constraints
python find_best_alpha_results.py --gnn1_variant gnn1_discomat_wo_constraints --gnn2_variant gnn2_discomat_wo_constraints --alphas 0.4 0.5 0.6 0.7 0.8

# DiSCoMat w/o caption
python find_best_alpha_results.py --gnn1_variant gnn1_discomat --gnn2_variant gnn2_discomat_wo_caption --alphas 0.4 0.5 0.6 0.7 0.8

# v-DiSCoMat
python find_best_alpha_results.py --gnn1_variant gnn1_vdiscomat --gnn2_variant gnn2_vdiscomat --alphas 0.4 0.5 0.6 0.7 0.8
