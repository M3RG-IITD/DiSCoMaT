from argparse import ArgumentParser
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score

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


def get_split_res(scc_res_file, non_scc_res_file):
    scc_res = pickle.load(open(os.path.join(table_dir, 'res_dir', scc_res_file), 'rb'))[split]
    non_scc_res = pickle.load(open(os.path.join(table_dir, 'res_dir', non_scc_res_file), 'rb'))[split]

    all_gold_tuples, all_pred_tuples = [], []
    all_gold_mids, all_pred_mids = [], []
    all_gold_table_type, all_pred_table_type = [], []
    violations = 0

    for i, pii_t_idx in enumerate(split_dict[split]):
        c = comp_data_dict[pii_t_idx]
        all_gold_tuples += get_gold_tuples(*pii_t_idx)
        all_gold_mids += c['gid_row_label'] + c['gid_col_label']
        if c['regex_table'] == 1:
            all_gold_table_type.append(0)
        elif c['sum_less_100'] == 0 and c['comp_table']:
            all_gold_table_type.append(1)
        elif c['sum_less_100'] == 1:
            all_gold_table_type.append(2)
        else:
            all_gold_table_type.append(3)

        scc_idx = scc_res['identifier'].index(pii_t_idx)
        non_scc_idx = non_scc_res['identifier'].index(pii_t_idx)
        assert scc_idx == non_scc_idx
        if scc_res['scc_pred'][scc_idx] == 1:
            all_pred_tuples += scc_res['tuples_pred'][scc_idx]
            all_pred_mids += scc_res['gid_pred_orig'][scc_idx]['row'] + scc_res['gid_pred_orig'][scc_idx]['col']
            all_pred_table_type.append(0)
            violations += cnt_violations(scc_res['gid_pred_orig'][scc_idx], scc_table=True)
        else:
            all_pred_tuples += non_scc_res['tuples_pred'][non_scc_idx]
            all_pred_mids += process_mids(non_scc_res['comp_gid_pred_orig'][non_scc_idx]['row'] + non_scc_res['comp_gid_pred_orig'][non_scc_idx]['col'])
            all_pred_table_type.append(non_scc_res['type_labels'][non_scc_idx])
            violations += cnt_violations(non_scc_res['comp_gid_pred_orig'][non_scc_idx], scc_table=False)

    table_type_acc = accuracy_score(all_gold_table_type, all_pred_table_type)
    mid_fscore = f1_score(all_gold_mids, all_pred_mids)
    tuple_metrics = get_tuples_metrics(all_gold_tuples, all_pred_tuples)
    mat_metrics = get_composition_metrics(all_gold_tuples, all_pred_tuples)

    return table_type_acc, mid_fscore, tuple_metrics, mat_metrics, violations


def compute_metrics(scc_variant, non_scc_variant):
    all_table_type_acc, all_mid_fscores, all_tuple_metrics, all_mat_metrics, all_violations = [], [], [], [], []
    for seed_1 in range(3):
        for seed_2 in range(3):
            table_type_acc, mid_fscore, tuple_metrics, mat_metrics, violations = \
            get_split_res(f'res_{scc_variant}_{seed_1}.pkl', f'res_{non_scc_variant}_{seed_2}.pkl')
            all_table_type_acc.append(table_type_acc)
            all_mid_fscores.append(mid_fscore)
            all_tuple_metrics.append(tuple_metrics)
            all_mat_metrics.append(mat_metrics)
            all_violations.append(violations)

    print('Table Type accuracy')
    print(round(np.mean(all_table_type_acc) * 100, 2), round(np.std(all_table_type_acc, ddof=1) * 100, 2))
    print()

    print('MID F1-score')
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


compute_metrics(args.gnn1_variant, args.gnn2_variant)
