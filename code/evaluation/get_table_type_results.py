from argparse import ArgumentParser
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from utils import get_tuples_metrics, get_composition_metrics, violation_funcs


parser = ArgumentParser()
parser.add_argument('--gnn1_variant', required=True, type=str)
parser.add_argument('--gnn2_variant', required=True, type=str)
parser.add_argument('--split', choices=['val', 'test'], default='test', type=str)
args = parser.parse_args()

table_dir = '../../data'

val_test_data = pickle.load(open(os.path.join(table_dir, 'val_test_data.pkl'), 'rb'))
split_dict = pickle.load(open(os.path.join(table_dir, 'train_val_test_split.pkl'), 'rb'))
comp_data_dict = {(c['pii'], c['t_idx']): c for c in val_test_data}
split = args.split


def get_gold_tuples(pii, t_idx):
    c = comp_data_dict[(pii, t_idx)]
    tuples = []
    for i in range(c['num_rows']):
        for j in range(c['num_cols']):
            if c['full_comp'][i][j] is None: continue
            for k in range(len(c['full_comp'][i][j])):
                prefix = f'{pii}_{t_idx}_{i}_{j}_{k}'
                for x in c['full_comp'][i][j][k]:
                    if x[2] == 0: continue
                    gid = prefix if x[0] is None else prefix + '_' + x[0]
                    tuples.append((gid, x[1], round(float(x[2]), 5), x[3]))
    return tuples


def cnt_violations(d: dict, scc_table=False):
    if scc_table:
        cnt_3 = (d['row'] + d['col']).count(1)
        return cnt_3 * (cnt_3 - 1) // 2
    return sum(f(d)[0] for f in violation_funcs.values())


def process_mids(l):
    return [1 if x == 3 else 0 for x in l]


def get_table_type_res(table_type, scc_res, non_scc_res):
    all_gold_tuples, all_pred_tuples = [], []
    all_gold_mids, all_pred_mids = [], []
    violations = 0
    
    if table_type == 'scc':
        for i, pii_t_idx in enumerate(split_dict[split]):
            c = comp_data_dict[pii_t_idx]
            if c['regex_table'] == 0: continue
            scc_idx = scc_res['identifier'].index(pii_t_idx)
            all_gold_tuples += get_gold_tuples(*pii_t_idx)
            all_pred_tuples += scc_res['true_scc_tuples_pred'][scc_idx]
            all_gold_mids += c['gid_row_label'] + c['gid_col_label']
            all_pred_mids += scc_res['true_scc_gid_pred_orig'][scc_idx]['row'] + scc_res['true_scc_gid_pred_orig'][scc_idx]['col']
            violations += cnt_violations(scc_res['true_scc_gid_pred_orig'][scc_idx], scc_table=True)
    
    elif table_type == 'mcc_ci':
        for i, pii_t_idx in enumerate(split_dict[split]):
            c = comp_data_dict[pii_t_idx]
            if c['regex_table'] == 1 or c['sum_less_100'] == 1 or not c['comp_table']: continue
            non_scc_idx = non_scc_res['identifier'].index(pii_t_idx)
            all_gold_tuples += get_gold_tuples(*pii_t_idx)
            all_pred_tuples += non_scc_res['tuples_pred'][non_scc_idx]
            all_gold_mids += c['gid_row_label'] + c['gid_col_label']
            all_pred_mids += process_mids(non_scc_res['comp_gid_pred_orig'][non_scc_idx]['row'] + non_scc_res['comp_gid_pred_orig'][non_scc_idx]['col'])
            violations += cnt_violations(non_scc_res['comp_gid_pred_orig'][non_scc_idx], scc_table=False)
    
    elif table_type == 'mcc_pi':
        for i, pii_t_idx in enumerate(split_dict[split]):
            c = comp_data_dict[pii_t_idx]
            if c['regex_table'] == 1 or c['sum_less_100'] == 0 or not c['comp_table']: continue
            non_scc_idx = non_scc_res['identifier'].index(pii_t_idx)
            all_gold_tuples += get_gold_tuples(*pii_t_idx)
            all_pred_tuples += non_scc_res['tuples_pred'][non_scc_idx]
            all_gold_mids += c['gid_row_label'] + c['gid_col_label']
            all_pred_mids += process_mids(non_scc_res['comp_gid_pred_orig'][non_scc_idx]['row'] + non_scc_res['comp_gid_pred_orig'][non_scc_idx]['col'])
            violations += cnt_violations(non_scc_res['comp_gid_pred_orig'][non_scc_idx], scc_table=False)
    else:
        raise NotImplementedError
    
    mid_fscore = f1_score(all_gold_mids, all_pred_mids)
    tuple_metrics = get_tuples_metrics(all_gold_tuples, all_pred_tuples)
    mat_metrics = get_composition_metrics(all_gold_tuples, all_pred_tuples)
    return mid_fscore, tuple_metrics, mat_metrics, violations


def compute_metrics(table_type, scc_variant, non_scc_variant):
    all_mid_fscores, all_tuple_metrics, all_mat_metrics, all_violations = [], [], [], []
    for seed in range(3):
        scc_res = pickle.load(open(os.path.join(table_dir, 'res_dir', f'res_{scc_variant}_{seed}.pkl'), 'rb'))[split]
        non_scc_res = pickle.load(open(os.path.join(table_dir, 'res_dir', f'res_{non_scc_variant}_{seed}.pkl'), 'rb'))
        if table_type.startswith('mcc'):
            non_scc_res = non_scc_res[f'{split}_{table_type}']

        mid_fscore, tuple_metrics, mat_metrics, violations = get_table_type_res(table_type, scc_res, non_scc_res)

        all_mid_fscores.append(mid_fscore)
        all_tuple_metrics.append(tuple_metrics)
        all_mat_metrics.append(mat_metrics)
        all_violations.append(violations)

    print('MID F1-Score')
    print(round(np.mean(all_mid_fscores) * 100, 2), round(np.std(all_mid_fscores, ddof=1) * 100, 2))
    print()

    tuple_df = pd.DataFrame(all_tuple_metrics)
    mat_df = pd.DataFrame(all_mat_metrics)

    mean_df = pd.concat([tuple_df.mean(), mat_df.mean()], axis=1).T
    mean_df.index = ['Tuple', 'Mat']
    print('mean')
    print(mean_df)

    std_df = pd.concat([tuple_df.std(ddof=1), mat_df.std(ddof=1)], axis=1).T
    std_df.index = ['Tuple', 'Mat']
    print('std')
    print(std_df)
    print()

    print('CV')
    print(round(np.mean(all_violations), 2), round(np.std(all_violations, ddof=1), 2))
    print()


for table_type in ['scc', 'mcc_ci', 'mcc_pi']:
    print(table_type.upper())
    compute_metrics(table_type, args.gnn1_variant, args.gnn2_variant)
