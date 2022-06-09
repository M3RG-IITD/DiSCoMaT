from collections import defaultdict, Counter
import os
import pickle
import re
import sys
sys.path.append('..')

import numpy as np
from torch.utils.data import Dataset


from regex_lib import parse_composition


table_dir = '../../data'
comp_data = pickle.load(open(os.path.join(table_dir, 'train_data.pkl'), 'rb')) + \
            pickle.load(open(os.path.join(table_dir, 'val_test_data.pkl'), 'rb'))
comp_data_dict = {(c['pii'], c['t_idx']): c for c in comp_data}
train_val_test_split = pickle.load(open(os.path.join(table_dir, 'train_val_test_split.pkl'), 'rb'))
splits = ['train', 'val', 'test']

for split in splits[1:]:
    for pii_t_idx in train_val_test_split[split]:
        c = comp_data_dict[pii_t_idx]
        if c['regex_table'] == 0:
            c['gid_row_label'], c['gid_col_label'] = [0] * c['num_rows'], [0] * c['num_cols']


def get_regex_feats(table):
    regex_feats = []
    for r in table:
        for cell in r:
            comp = parse_composition(cell)
            if len(comp) == 0 or len(comp[0][0]) == 1:
                regex_feats.append(1)
            else:
                regex_feats.append(2)
    return regex_feats


def get_max_freq_feat(table):
    max_freq_feat = []
    for r in table:
        cnt = Counter()
        for cell in r:
            if cell: cnt[cell] += 1
        max_freq_feat.append(cnt.most_common(1)[0][1] if cnt else 1)
    for j in range(len(table[0])):
        cnt = Counter()
        for i in range(len(table)):
            cell = table[i][j]
            if cell: cnt[cell] += 1
        max_freq_feat.append(cnt.most_common(1)[0][1] if cnt else 1)
    return max_freq_feat


class TableDataset(Dataset):
    def __init__(self, data):
        self.inp = data

    def __getitem__(self, idx):
        return self.inp[idx]

    def __len__(self):
        return len(self.inp)


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


def get_gold_tuples(pii, t_idx):
    tuples = []
    c = comp_data_dict[(pii, t_idx)]
    if c['regex_table'] == 0:
        return tuples
    for i in range(c['num_rows']):
        for j in range(c['num_cols']):
            if c['full_comp'][i][j] is None: continue
            for k in range(len(c['full_comp'][i][j])):
                for x in c['full_comp'][i][j][k]:
                    if x[2] == 0: continue
                    prefix = f'{pii}_{t_idx}_{i}_{j}_{k}'
                    if x[0] is None:
                        xx = (prefix, x[1], round(x[2], 5), x[3])
                    else:
                        xx = (prefix + '_' + x[0], x[1], round(x[2], 5), x[3])
                    tuples.append(xx)
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
    return {'precision': prec, 'recall': rec, 'fscore': fscore}


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
    return {'precision': prec, 'recall': rec, 'fscore': fscore}


def cnt_3_3_violations(d: dict):
    r, c = len(d['row']), len(d['col'])
    cnt_3 = (d['row'] + d['col']).count(1)
    return cnt_3 * (cnt_3 - 1) // 2, (r + c) * (r + c - 1) // 2

