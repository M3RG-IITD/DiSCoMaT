#!/bin/sh

# DiSCoMat
python get_table_type_results.py --gnn1_variant gnn1_discomat --gnn2_variant gnn2_discomat

# DiSCoMat w/o features
python get_table_type_results.py --gnn1_variant gnn1_discomat_wo_features --gnn2_variant gnn2_discomat_wo_features

# DiSCoMat w/o constraints
python get_table_type_results.py --gnn1_variant gnn1_discomat_wo_constraints --gnn2_variant gnn2_discomat_wo_constraints

# DiSCoMat w/o caption
python get_table_type_results.py --gnn1_variant gnn1_discomat --gnn2_variant gnn2_discomat_wo_caption

# v-DiSCoMat
python get_table_type_results.py --gnn1_variant gnn1_vdiscomat --gnn2_variant gnn2_vdiscomat
