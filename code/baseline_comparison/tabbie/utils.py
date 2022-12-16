from collections import defaultdict
import os
import pickle
import re

import numpy as np
from torch.utils.data import Dataset
import pdb


table_dir = './data'
comp_data = pickle.load(open(os.path.join(table_dir, 'train_data.pkl'), 'rb')) + \
            pickle.load(open(os.path.join(table_dir, 'val_test_data.pkl'), 'rb'))
#pdb.set_trace()

train_val_test_split = pickle.load(open(os.path.join(table_dir, 'train_val_test_split.pkl'), 'rb'))
comp_data_dict = {(c['pii'], c['t_idx']): c for c in comp_data}

splits = ['train', 'val', 'test']

for split in splits[:3]:
    train_val_test_split[split] = [x for x in train_val_test_split[split] if comp_data_dict[x]['regex_table'] == 0 and comp_data_dict[x]['sum_less_100'] == 0]

for split in splits:
    for pii_t_idx in train_val_test_split[split]:
        c = comp_data_dict[pii_t_idx]
        if sum(c['gid_row_label']) == 1:
            c['row_label'][c['gid_row_label'].index(1)] = 3
        elif sum(c['gid_col_label']) == 1:
            c['col_label'][c['gid_col_label'].index(1)] = 3


class TableDataset(Dataset):
    def __init__(self, data):
        self.inp = data

    def __getitem__(self, idx):
        return self.inp[idx]

    def __len__(self):
        return len(self.inp)


def rectify_comp_labels(row_labels, col_labels):
    row_1, row_2, col_1, col_2 = row_labels.count(1), row_labels.count(2), col_labels.count(1), col_labels.count(2)
    if row_1 + col_1 == 0 or row_2 + col_2 == 0 or row_1 + row_2 == 0 or col_1 + col_2 == 0:
        return [0] * len(row_labels), [0] * len(col_labels)
    if (row_2 + col_1 == 0) or (row_1 + col_2 == 0):
        return row_labels, col_labels
    if row_1 == 0:
        return row_labels, [0 if c == 2 else c for c in col_labels]
    if row_2 == 0:
        return row_labels, [0 if c == 1 else c for c in col_labels]
    if col_1 == 0:
        return [0 if r == 2 else r for r in row_labels], col_labels
    if col_2 == 0:
        return [0 if r == 1 else r for r in row_labels], col_labels
    if row_1 > row_2 and col_2 > col_1:
        return [0 if r == 2 else r for r in row_labels], [0 if c == 1 else c for c in col_labels]
    if row_2 > row_1 and col_1 > col_2:
        return [0 if r == 1 else r for r in row_labels], [0 if c == 2 else c for c in col_labels]
    return [0] * len(row_labels), [0] * len(col_labels)


all_elements = pickle.load(open(os.path.join(table_dir, 'elements_compounds.pkl'), 'rb'))['elements']
comp_num_pattern = r'(\d+\.\d+|\d+/\d+|\d+)'
ele_num = r'((' + '|'.join(all_elements) + r')' + comp_num_pattern + r'?)'
many_ele_num = r'(' + ele_num + r')+'
ele_comp_pattern = r'(((\(' + many_ele_num + '\)' + comp_num_pattern + r')|(' + many_ele_num + r'))+)'
ele_comp_pattern = r'(?:^|\W)(' + ele_comp_pattern + r'|Others?)(?:\W|$)'
num = r'(\d+\.\d+|\d+)'

mol_regex = re.compile(r'mol|at\.?\s*\%', re.IGNORECASE)
wt_regex = re.compile(r'mass\s*(%|percent)|weight\s*(%|percent)|wt\.?\s*\%', re.IGNORECASE)


def find_mol_wt_in_text(text):
    if re.search(mol_regex, text):
        return 'mol'
    if re.search(wt_regex, text):
        return 'wt'
    return ''


def pred_cell_mol_wt(table_dict, i, j):
    for k in reversed(range(i+j+1)): # increasing L1 distance from (i, j)
        for i_ in range(max(k-j, 0), min(k,i)+1):
            j_ = k - i_
            pred = find_mol_wt_in_text(table_dict['act_table'][i_][j_])
            if pred: return pred

    pred = find_mol_wt_in_text(table_dict['caption'])
    if pred: return pred
    return 'mol'


def get_sum_100_pred_tuples(pii, t_idx, edges, comp_gid_pred):
    c = comp_data_dict[(pii, t_idx)]
    row_labels, col_labels = comp_gid_pred['row'], comp_gid_pred['col']
    tuples = []
    if sum(row_labels) == 0 or sum(col_labels) == 0: return tuples
    
    def proc_edges(e):
        return (e[0][0] * c['num_cols'] + e[0][1], e[1][0] * c['num_cols'] + e[1][1])
    
    edges = np.array(list(map(proc_edges, edges)))
    if 1 in row_labels:
        comp_cols = [j for j in range(c['num_cols']) if col_labels[j] == 2]
        gid_cols = [j for j in range(c['num_cols']) if col_labels[j] == 3]
        gid_col = gid_cols[0] if gid_cols else None
        for i in range(c['num_rows']):
            if row_labels[i] != 1: continue
            g_id = f'{pii}_{t_idx}_{i}_{comp_cols[0]}_0'
            if gid_col is not None and c['act_table'][i][gid_col]:
                g_id += '_' + c['act_table'][i][gid_col]
            for j in comp_cols:
                s = re.search(num, c['act_table'][i][j])
                if s is None or float(s.group()) == 0: continue
                src = i * c['num_cols'] + j
                src_edges = edges[edges[:, 0] == src]
                dst = src_edges[0][1]
                dst_contents = c['act_table'][dst // c['num_cols']][dst % c['num_cols']]
                ele_comp = re.findall(ele_comp_pattern, dst_contents)
                if len(ele_comp) == 0: continue
                tuples.append((g_id, ele_comp[0][0] if ele_comp[0][0] != 'Other' else 'Others', float(s.group()), pred_cell_mol_wt(c, i, j)))
    else:
        comp_rows = [i for i in range(c['num_rows']) if row_labels[i] == 2]
        gid_rows = [i for i in range(c['num_rows']) if row_labels[i] == 3]
        gid_row = gid_rows[0] if gid_rows else None
        for j in range(c['num_cols']):
            if col_labels[j] != 1: continue
            g_id = f'{pii}_{t_idx}_{comp_rows[0]}_{j}_0'
            if gid_row is not None and c['act_table'][gid_row][j]:
                g_id += '_' + c['act_table'][gid_row][j]
            for i in comp_rows:
                s = re.search(num, c['act_table'][i][j])
                if s is None or float(s.group()) == 0: continue
                src = i * c['num_cols'] + j
                src_edges = edges[edges[:, 0] == src]
                dst = src_edges[0][1]
                dst_contents = c['act_table'][dst // c['num_cols']][dst % c['num_cols']]
                ele_comp = re.findall(ele_comp_pattern, dst_contents)
                if len(ele_comp) == 0: continue
                tuples.append((g_id, ele_comp[0][0] if ele_comp[0][0] != 'Other' else 'Others', float(s.group()), pred_cell_mol_wt(c, i, j)))
    return tuples


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


def get_tuples_metrics(gold_tuples, pred_tuples):
    prec = 0
    for p in pred_tuples:
        if p in gold_tuples:
            prec += 1
    if len(pred_tuples) > 0:
        prec /= len(pred_tuples)
    else:
        prec = 0.0
    rec = 0
    for g in gold_tuples:
        if g in pred_tuples:
            rec += 1
    rec /= len(gold_tuples)
    fscore = 2 * prec * rec / (prec + rec) if (prec + rec > 0) else 0.0
    metrics = {'precision': prec, 'recall': rec, 'fscore': fscore}
    metrics = {m: round(v * 100, 2) for m, v in metrics.items()}
    return metrics


def get_composition_metrics(gold_tuples, pred_tuples):
    gold_comps, pred_comps = defaultdict(set), defaultdict(set)
    for g in gold_tuples:
        gold_comps[g[0]].add((g[1], g[2], g[3]))
    for p in pred_tuples:
        pred_comps[p[0]].add((p[1], p[2], p[3]))

    prec = 0
    for p, v in pred_comps.items():
        if p in gold_comps and gold_comps[p] == v:
            prec += 1
    if len(pred_comps) > 0:
        prec /= len(pred_comps)
    else:
        prec = 0.0
    rec = 0
    for g, v in gold_comps.items():
        if g in pred_comps and pred_comps[g] == v:
            rec += 1
    rec /= len(gold_comps)
    fscore = 2 * prec * rec / (prec + rec) if (prec + rec > 0) else 0.0
    metrics = {'precision': prec, 'recall': rec, 'fscore': fscore}
    metrics = {m: round(v * 100, 2) for m, v in metrics.items()}
    return metrics


def cnt_1_2_violations(d: dict):
    r, c = len(d['row']), len(d['col'])
    return d['row'].count(1) * d['col'].count(1) + d['row'].count(2) * d['col'].count(2), 2 * r * c


def cnt_1_3_violations(d: dict):
    r, c = len(d['row']), len(d['col'])
    return d['row'].count(1) * d['row'].count(3) + d['col'].count(1) * d['col'].count(3), r * (r - 1) + c * (c - 1)


def cnt_2_3_violations(d: dict):
    r, c = len(d['row']), len(d['col'])
    return d['row'].count(2) * d['col'].count(3) + d['row'].count(3) * d['col'].count(2), 2 * r * c


def cnt_3_3_violations(d: dict):
    r, c = len(d['row']), len(d['col'])
    cnt_3 = (d['row'] + d['col']).count(3)
    return cnt_3 * (cnt_3 - 1) // 2, (r + c) * (r + c - 1) // 2


violation_funcs = {
    '1_2_violations': cnt_1_2_violations,
    '1_3_violations': cnt_1_3_violations,
    '2_3_violations': cnt_2_3_violations,
    '3_3_violations': cnt_3_3_violations,
}

