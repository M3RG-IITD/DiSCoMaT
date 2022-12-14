from argparse import ArgumentParser
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

from utils import get_tuples_metrics, get_composition_metrics, violation_funcs


parser = ArgumentParser()
parser.add_argument('--gnn1_variant', required=True, type=str)
parser.add_argument('--gnn2_variant', required=True, type=str)
parser.add_argument('--alphas', nargs='+', required=True, type=float)
args = parser.parse_args()

# alphas = [0.4, 0.5, 0.6, 0.7, 0.8]
alphas = args.alphas
print("Alphas:", alphas)

table_dir = '../../data'

val_test_data = pickle.load(open(os.path.join(table_dir, 'val_test_data.pkl'), 'rb'))
split_dict = pickle.load(open(os.path.join(table_dir, 'train_val_test_split.pkl'), 'rb'))
comp_data_dict = {(c['pii'], c['t_idx']): c for c in val_test_data}


def get_gold_tuples(pii, t_idx):
    '''
    Function for obtaining gold tuples using paper PII and table index
    '''
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

# split = args.split
def get_split_res(scc_res_file, non_scc_res_file, split):
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
    table_type_fscore = f1_score(all_gold_table_type, all_pred_table_type, average='weighted')
    table_type_precision = precision_score(all_gold_table_type, all_pred_table_type, average='weighted')
    table_type_recall = recall_score(all_gold_table_type, all_pred_table_type, average='weighted')

    mid_fscore = f1_score(all_gold_mids, all_pred_mids)
    mid_precision = precision_score(all_gold_mids, all_pred_mids)
    mid_recall = recall_score(all_gold_mids, all_pred_mids)
    mid_accuracy = accuracy_score(all_gold_mids, all_pred_mids)

    tuple_metrics = get_tuples_metrics(all_gold_tuples, all_pred_tuples)
    mat_metrics = get_composition_metrics(all_gold_tuples, all_pred_tuples)

    return (table_type_acc, table_type_precision, table_type_recall, table_type_fscore), (mid_fscore, mid_precision, mid_recall, mid_accuracy), tuple_metrics, mat_metrics, violations


def compute_metrics(scc_variant, non_scc_variant):
    (all_table_type_acc, all_table_type_fscore, all_table_type_precision, all_table_type_recall) , (all_mid_fscore, all_mid_precision, all_mid_recall, all_mid_accuracy), all_tuple_metrics, all_mat_metrics, all_violations = ([], [], [], []), ([], [], [], []), [], [], []

    for seed_1 in range(3):
        for seed_2 in range(3):

            best_alpha = None
            best_score = None

            for alpha in alphas:
                # find best alpha for this seed pair on val set
                print(f'seed1 {seed_1} seed2 {seed_2} alpha {alpha}')
                (table_type_acc, table_type_precision, table_type_recall, table_type_fscore), (mid_fscore, mid_precision, mid_recall, mid_accuracy), tuple_metrics, mat_metrics, violations = \
                get_split_res(f'res_{scc_variant}_mid_alpha_{alpha}_{seed_1}.pkl', f'res_{non_scc_variant}_mid_alpha_{alpha}_{seed_2}.pkl', 'val')
                if best_score == None:
                    best_score = mid_fscore
                    best_alpha = alpha
                else:
                    if best_score < mid_fscore:
                        best_score = mid_fscore
                        best_alpha = alpha

            print(f'best alpha {best_alpha}')
            (table_type_acc, table_type_precision, table_type_recall, table_type_fscore), (mid_fscore, mid_precision, mid_recall, mid_accuracy), tuple_metrics, mat_metrics, violations = \
                get_split_res(f'res_{scc_variant}_mid_alpha_{best_alpha}_{seed_1}.pkl', f'res_{non_scc_variant}_mid_alpha_{best_alpha}_{seed_2}.pkl', 'test')
            all_table_type_acc.append(table_type_acc)
            all_table_type_fscore.append(table_type_fscore)
            all_table_type_precision.append(table_type_precision)
            all_table_type_recall.append(table_type_recall)
            all_mid_fscore.append(mid_fscore)
            all_mid_precision.append(mid_precision)
            all_mid_recall.append(mid_recall)
            all_mid_accuracy.append(mid_accuracy)
            all_tuple_metrics.append(tuple_metrics)
            all_mat_metrics.append(mat_metrics)
            all_violations.append(violations)

    print('Table Type Accuracy')
    print(round(np.mean(all_table_type_acc) * 100, 2), round(np.std(all_table_type_acc, ddof=1) * 100, 2))
    print()
    print('Table Type Precision')
    print([round(x, 2) for x in all_table_type_precision])
    print(round(np.mean(all_table_type_precision) * 100, 2), round(np.std(all_table_type_precision, ddof=1) * 100, 2))
    print('Table Type Recall')
    print([round(x, 2) for x in all_table_type_recall])
    print(round(np.mean(all_table_type_recall) * 100, 2), round(np.std(all_table_type_recall, ddof=1) * 100, 2))
    print()
    print('Table Type F1-score')
    print([round(x, 2) for x in all_table_type_fscore])
    print(round(np.mean(all_table_type_fscore) * 100, 2), round(np.std(all_table_type_fscore, ddof=1) * 100, 2))
    print()

    print('MID F1-score')
    print(round(np.mean(all_mid_fscore) * 100, 2), round(np.std(all_mid_fscore, ddof=1) * 100, 2))
    print()
    print('MID Precision')
    print([round(x, 2) for x in all_mid_precision])
    print(round(np.mean(all_mid_precision) * 100, 2), round(np.std(all_mid_precision, ddof=1) * 100, 2))
    print('MID Recall')
    print([round(x, 2) for x in all_mid_recall])
    print(round(np.mean(all_mid_recall) * 100, 2), round(np.std(all_mid_recall, ddof=1) * 100, 2))
    print()
    print('MID Accuracy')
    print([round(x, 2) for x in all_mid_accuracy])
    print(round(np.mean(all_mid_accuracy) * 100, 2), round(np.std(all_mid_accuracy, ddof=1) * 100, 2))
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
