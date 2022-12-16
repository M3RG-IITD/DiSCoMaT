import os
import pickle

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, precision_recall_fscore_support
import torch
from torch import nn, Tensor, LongTensor
import torch.nn.functional as F

from sklearn.metrics import classification_report, precision_recall_fscore_support
import random
import pdb

from utils import *


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
#device = torch.device('cpu')
print(device)


def get_all_pair(ar):
    return np.array(np.meshgrid(ar, ar)).T.reshape(-1, 2)


def get_edges(r, c):
    edges = np.empty((0, 2), dtype=int)
    row_edges = get_all_pair(np.arange(c))
    for i in range(r):
        edges = np.concatenate((edges, row_edges + i * c), axis=0)
    col_edges = get_all_pair(np.arange(0, r * c, c))
    for i in range(c):
        edges = np.concatenate((edges, col_edges + i), axis=0)
    edges = np.unique(edges, axis=0)
    table_edges = LongTensor(edges[np.lexsort((edges[:, 1], edges[:, 0]))])
    assert len(table_edges) == r * c * (r + c - 1)
    return table_edges


def create_edge_labels(all_edges, edge_list):
    df = pd.DataFrame(all_edges.tolist())
    df['merge'] = list(zip(df[0], df[1]))
    edge_labels = LongTensor(df['merge'].isin(edge_list))
    return edge_labels


def get_edge_gid_pred_labels(comp_gid_labels, comp_gid_logits, edge_logits, num_rows: int, num_cols: int):
    row_col_labels = [0 if c == 3 else c for c in comp_gid_labels]
    row_labels, col_labels = rectify_comp_labels(row_col_labels[:num_rows], row_col_labels[-num_cols:])
    r, c = Tensor(row_labels).unsqueeze(1), Tensor(col_labels).unsqueeze(0)
    comp_cells = np.where((r * c).view(-1) == 2)[0]

    if sum(row_labels + col_labels) > 0:
        if 1 in row_labels:
            assert 2 in col_labels
            if 3 in comp_gid_labels[-num_cols:]:
                col_gid_probs = F.softmax(comp_gid_logits[-num_cols:], dim=1)
                idx = np.where(col_gid_probs[:, 3] == col_gid_probs[col_gid_probs.argmax(1) == 3, 3].max())[0][0]
                col_labels[idx] = 3
        else:
            assert 1 in col_labels and 2 in row_labels
            if 3 in comp_gid_labels[:num_rows]:
                row_gid_probs = F.softmax(comp_gid_logits[:num_rows], dim=1)
                idx = np.where(row_gid_probs[:, 3] == row_gid_probs[row_gid_probs.argmax(1) == 3, 3].max())[0][0]
                row_labels[idx] = 3

    df = pd.DataFrame(get_edges(num_rows, num_cols).tolist())
    df.columns = ['src', 'dst']
    df['wt'] = edge_logits
    idx = (df.groupby('src')['wt'].transform(max) == df['wt']) & df['src'].isin(comp_cells)
    df.drop('wt', inplace=True, axis=1)
    edges = df[idx].applymap(lambda x: (x // num_cols, x % num_cols)).values.tolist()
    return row_labels + col_labels, list(idx.astype(int).values), edges


def get_batch_edge_gid_pred_labels(comp_gid_logits, edge_logits, num_rows: list, num_cols: list):
    base_comp_gid, base_edge = 0, 0
    pred_edge_labels, pred_edges, pred_row_col_gid_labels = [], [], []
    comp_gid_labels = comp_gid_logits.argmax(1).tolist()
    for r, c in zip(num_rows, num_cols):
        num_comp_labels, num_edge_logits = r + c, r * c * (r + c - 1)
        row_col_gid_labels, edge_labels, edges = get_edge_gid_pred_labels(
            comp_gid_labels[base_comp_gid:base_comp_gid+num_comp_labels], 
            comp_gid_logits[base_comp_gid:base_comp_gid+num_comp_labels], 
            edge_logits[base_edge:base_edge+num_edge_logits], r, c)
        pred_edge_labels += edge_labels
        pred_edges.append(edges)
        pred_row_col_gid_labels += row_col_gid_labels
        base_comp_gid += num_comp_labels
        base_edge += num_edge_logits
    return pred_row_col_gid_labels, pred_edge_labels, pred_edges


def eval_model(split, debug=False):
    datasets = dict()
    identifier = []
    y_comp_true, y_comp_pred, ret_comp_pred = [], [], []
    y_edge_true, y_edge_pred, ret_edge_pred = [], [], []
    base = 0
    batch_all_edges, batch_table_edges, batch_edge_labels = [], [], []
    datasets[split] = TableDataset([comp_data_dict[pii_t_idx] for pii_t_idx in train_val_test_split[split]])
    limit = len(datasets[split])
    
    gold_tuples = dict()
    gold_tuples[split] = []
    if split!='train':
        for pii, t_idx in train_val_test_split[split]:
            gold_tuples[split] += get_gold_tuples(pii, t_idx) 
    
    for i in range(limit):
        x = datasets[split][i]
        table = x['act_table']
        pii = x['pii']
        t_idx = x['t_idx']
        new_name = pii + '__' + str(t_idx)
        
        comp_gid_logits = torch.load(f'./row_col_gid_logit_17_ada/{new_name}.pt', map_location=device)
        comp_gid_labels = x['row_label'] + x['col_label']
        assert comp_gid_logits.size()[0] == len(comp_gid_labels)
        edge_logits = torch.load(f'./edge_logit_17_ada/{new_name}.pt', map_location=device)
        #edge_labels = x['edge_list']
        y_comp_true += comp_gid_labels
        
        act_table_edges = get_edges(x['num_rows'], x['num_cols'])
        batch_edge_labels = create_edge_labels(act_table_edges, x['edge_list'])
        
        y_edge_true += batch_edge_labels
        y_comp_pred += comp_gid_logits.argmax(1).cpu().detach().tolist()
        num_rows, num_cols = [x['num_rows']], [x['num_cols']]
        #pdb.set_trace()
        pred_comp_gid_labels, pred_edge_labels, batch_pred_edges = get_batch_edge_gid_pred_labels(comp_gid_logits.cpu().detach(), edge_logits.cpu().detach(), num_rows, num_cols)
        #pdb.set_trace()
        assert len(batch_edge_labels)==len(pred_edge_labels)
        y_edge_pred += pred_edge_labels
        
        ret_edge_pred += batch_pred_edges
        
        base_comp_gid = 0
        identifier.append((x['pii'], x['t_idx']))
        comp_dict = dict()
        comp_dict['row'] = pred_comp_gid_labels[base_comp_gid:base_comp_gid+x['num_rows']]
        base_comp_gid += x['num_rows']
        comp_dict['col'] = pred_comp_gid_labels[base_comp_gid:base_comp_gid+x['num_cols']]
        base_comp_gid += x['num_cols']
        ret_comp_pred.append(comp_dict)
        
    #pdb.set_trace()    
    comp_gid_metrics = pd.DataFrame(classification_report(y_comp_true, y_comp_pred, labels=[1, 2, 3], output_dict=True))
    prec, recall, fscore, _ = precision_recall_fscore_support(y_edge_true, y_edge_pred, average='binary')
    edge_metrics = {'precision': prec, 'recall': recall, 'fscore': fscore}
    
    if split=='train':
        return identifier, (comp_gid_metrics.iloc[:3, :3], y_comp_pred, ret_comp_pred), (edge_metrics, ret_edge_pred), \
        (0, 0), 0
    else:
        ret_tuples_pred, all_tuples_pred = [], []
        for (pii, t_idx), edges, comp_gid_pred in zip(identifier, ret_edge_pred, ret_comp_pred):
            tuples_pred = get_sum_100_pred_tuples(pii, t_idx, edges, comp_gid_pred) #where is this func?
            ret_tuples_pred.append(tuples_pred)
            all_tuples_pred += tuples_pred
    
        tuple_metrics = get_tuples_metrics(gold_tuples[split], all_tuples_pred)
        composition_metrics = get_composition_metrics(gold_tuples[split], all_tuples_pred)

    if not debug:
        return comp_gid_metrics.iloc[:3, :3], edge_metrics, tuple_metrics, composition_metrics
    else:
        return identifier, (comp_gid_metrics.iloc[:3, :3], y_comp_pred, ret_comp_pred), (edge_metrics, ret_edge_pred), \
        (tuple_metrics, ret_tuples_pred), composition_metrics


res = dict()
for s in splits[2:]:
    
    res[s] = dict()
    res[s]['identifier'], (res[s]['comp_gid_stats'], res[s]['comp_gid_pred_orig'], res[s]['comp_gid_pred']), \
    (res[s]['edge_stats'], res[s]['edge_pred']), (res[s]['tuple_metrics'], res[s]['tuples_pred']), \
    res[s]['composition_metrics'] = eval_model(s, debug=True)

    print(f'{s} comp F1:\n', res[s]['comp_gid_stats'])
    print(f'{s} edge F1:\n', res[s]['edge_stats'])
    print(f'{s} tuple metrics:\n', res[s]['tuple_metrics'])
    print(f'{s} compostion metrics:\n', res[s]['composition_metrics'])

    comp_gid_pred_orig = []
    base = 0
    for pii_t_idx in res[s]['identifier']:
        c = comp_data_dict[pii_t_idx]
        d = dict()
        d['row'] = res[s]['comp_gid_pred_orig'][base:base+c['num_rows']]
        base += c['num_rows']
        d['col'] = res[s]['comp_gid_pred_orig'][base:base+c['num_cols']]
        base += c['num_cols']
        comp_gid_pred_orig.append(d)
    res[s]['comp_gid_pred_orig'] = comp_gid_pred_orig

    for n, f in violation_funcs.items():
        violations, total = 0, 0
        for table in res[s]['comp_gid_pred_orig']:
            v, t = f(table)
            violations += v
            total += t
        print(f'{s} {n} violations: {violations}/{total}')
        res[s][n] = violations
        
    

# if args.res_file:
#     os.makedirs(os.path.join(table_dir, 'res_dir'), exist_ok=True)
    pickle.dump(res, open('./results_17_adapted.pkl', 'wb'))