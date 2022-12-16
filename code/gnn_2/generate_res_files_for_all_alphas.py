import math
import multiprocessing as mp
import os
import pickle

from argparse import ArgumentParser
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, precision_recall_fscore_support
from sklearn.utils.class_weight import compute_class_weight
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import set_seed, get_linear_schedule_with_warmup

from gnn_2_model import GNN_2_Model as Model
from utils import *


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print('using device:', device)


parser = ArgumentParser()
parser.add_argument('--seeds', nargs='+', required=True, type=int)
parser.add_argument('--alphas', nargs='+', required=True, type=float)
parser.add_argument('--hidden_layer_sizes', nargs='+', required=True, type=int)
parser.add_argument('--num_heads', nargs='+', required=True, type=int)
parser.add_argument('--use_max_freq_feat', action='store_true')
parser.add_argument('--max_freq_emb_size', required=False, default=128, type=int)
parser.add_argument('--add_constraint', action='store_true')
parser.add_argument('--use_caption', action='store_true')
parser.add_argument('--model_variant', required=True, type=str)
args = parser.parse_args()
print(args)

# alphas = [0.4, 0.5, 0.6, 0.7, 0.8]
alphas = args.alphas
print("Alphas:", alphas)

lm_name = 'm3rg-iitd/matscibert'
cache_dir = os.path.join(table_dir, '.cache')

if args.use_max_freq_feat:
    for c in comp_data:
        c['max_freq_feat'] = get_max_freq_feat(c['act_table'])

torch.set_deterministic(True)
torch.backends.cudnn.benchmark = False

datasets = dict()
for split in splits:
    datasets[split] = TableDataset([comp_data_dict[pii_t_idx] for pii_t_idx in train_val_test_split[split]])

batch_size = 8
num_workers = mp.cpu_count()
loaders = dict()
for split in splits:
    loaders[split] = DataLoader(datasets[split], batch_size=batch_size, shuffle=(split == 'train'), \
    num_workers=num_workers, collate_fn=lambda x: x)

all_train_comp_labels = []
for x in datasets['train']:
    all_train_comp_labels += x['row_label'] + x['col_label']

all_train_edge_labels = []
for x in datasets['train']:
    table_edges = Model.get_edges(x['num_rows'], x['num_cols'])
    all_train_edge_labels += Model.create_edge_labels(table_edges, x['edge_list']).tolist()

model_args = {
    'hidden_layer_sizes': args.hidden_layer_sizes,
    'num_heads': args.num_heads,
    'lm_name': lm_name,
    'cache_dir': cache_dir,
    'use_max_freq_feat': args.use_max_freq_feat,
    'max_freq_emb_size': args.max_freq_emb_size,
    'add_constraint': args.add_constraint,
    'use_caption': args.use_caption,
}
model = Model(model_args).to(device)

gold_tuples = dict()
for split in splits[1:]:
    gold_tuples[split] = []
    for pii, t_idx in train_val_test_split[split]:
        gold_tuples[split] += get_gold_tuples(pii, t_idx)


def get_edge_gid_pred_labels(comp_gid_labels, comp_gid_logits, edge_logits, num_rows: int, num_cols: int, mid_alpha):
    row_col_labels = [0 if c == 3 else c for c in comp_gid_labels]
    row_labels, col_labels = rectify_comp_labels(row_col_labels[:num_rows], row_col_labels[-num_cols:])
    assert row_labels.count(1) * col_labels.count(1) + row_labels.count(2) * col_labels.count(2) == 0
    r, c = Tensor(row_labels).unsqueeze(1), Tensor(col_labels).unsqueeze(0)
    comp_cells = np.where((r * c).view(-1) == 2)[0]

    if sum(row_labels + col_labels) > 0:
        if 1 in row_labels:
            assert 2 in col_labels
            if 3 in comp_gid_labels[-num_cols:]:
                col_gid_probs = F.softmax(comp_gid_logits[-num_cols:], dim=1)
                probs_max = col_gid_probs.max(1)[0].cpu().detach().tolist()
                if col_gid_probs[col_gid_probs.argmax(1) == 3, 3].max() > mid_alpha:
                    idx = np.where(col_gid_probs[:, 3] == col_gid_probs[col_gid_probs.argmax(1) == 3, 3].max())[0][0]
                    col_labels[idx] = 3
                            
        else:
            assert 1 in col_labels and 2 in row_labels
            if 3 in comp_gid_labels[:num_rows]:
                row_gid_probs = F.softmax(comp_gid_logits[:num_rows], dim=1)
                if row_gid_probs[row_gid_probs.argmax(1) == 3, 3].max() > mid_alpha:
                    idx = np.where(row_gid_probs[:, 3] == row_gid_probs[row_gid_probs.argmax(1) == 3, 3].max())[0][0]
                    row_labels[idx] = 3

    df = pd.DataFrame(Model.get_edges(num_rows, num_cols).tolist())
    df.columns = ['src', 'dst']
    df['wt'] = edge_logits
    idx = (df.groupby('src')['wt'].transform(max) == df['wt']) & df['src'].isin(comp_cells)
    df.drop('wt', inplace=True, axis=1)
    edges = df[idx].applymap(lambda x: (x // num_cols, x % num_cols)).values.tolist()
    return row_labels + col_labels, list(idx.astype(int).values), edges


def get_batch_edge_gid_pred_labels(comp_gid_logits, edge_logits, num_rows: list, num_cols: list, mid_alpha):
    base_comp_gid, base_edge = 0, 0
    pred_edge_labels, pred_edges, pred_row_col_gid_labels = [], [], []
    comp_gid_labels = comp_gid_logits.argmax(1).tolist()
    for r, c in zip(num_rows, num_cols):
        num_comp_labels, num_edge_logits = r + c, r * c * (r + c - 1)
        row_col_gid_labels, edge_labels, edges = get_edge_gid_pred_labels(
            comp_gid_labels[base_comp_gid:base_comp_gid+num_comp_labels], 
            comp_gid_logits[base_comp_gid:base_comp_gid+num_comp_labels], 
            edge_logits[base_edge:base_edge+num_edge_logits], r, c, mid_alpha)
        pred_edge_labels += edge_labels
        pred_edges.append(edges)
        pred_row_col_gid_labels += row_col_gid_labels
        base_comp_gid += num_comp_labels
        base_edge += num_edge_logits
    return pred_row_col_gid_labels, pred_edge_labels, pred_edges


def eval_model(split, mid_alpha, debug=False):
    model.eval()
    identifier = []
    y_comp_true, y_comp_pred, ret_comp_pred = [], [], []
    y_edge_true, y_edge_pred, ret_edge_pred = [], [], []

    with torch.no_grad():

        tepoch = tqdm(loaders[split], unit='batch')
        for batch_data in tepoch:
            tepoch.set_description(f'{split} mode')
            (comp_gid_logits, comp_gid_labels), (edge_logits, edge_labels) = model(batch_data)
            y_comp_true += comp_gid_labels.cpu().detach().tolist()
            y_edge_true += edge_labels.cpu().detach().tolist()
            
            comp_probs = F.softmax(comp_gid_logits, dim=1)
            batch_y_comp_pred = comp_gid_logits.argmax(1).cpu().detach().tolist()
            batch_y_comp_pred_1 = comp_probs.argmax(1).cpu().detach().tolist()
            assert batch_y_comp_pred == batch_y_comp_pred_1
            for i, (pred_probs, pred_label) in enumerate(zip(comp_probs, batch_y_comp_pred)):
                # if labelled gid but logit value after softmax is less than mid_alpha then label 0
                if pred_label == 3:
                    assert comp_probs[i, 3] == pred_probs[3]
                    assert comp_gid_logits[i, 3] >= comp_gid_logits[i, 0] and comp_gid_logits[i, 3] >= comp_gid_logits[i, 1] and comp_gid_logits[i, 3] >= comp_gid_logits[i, 2]
                    if pred_probs[3] < mid_alpha:
                        batch_y_comp_pred[i] = 0
            y_comp_pred += batch_y_comp_pred

            num_rows, num_cols = [x['num_rows'] for x in batch_data], [x['num_cols'] for x in batch_data]
            pred_comp_gid_labels, pred_edge_labels, batch_pred_edges = get_batch_edge_gid_pred_labels(
                comp_gid_logits.cpu().detach(), edge_logits.cpu().detach(), num_rows, num_cols, mid_alpha)
            y_edge_pred += pred_edge_labels
            ret_edge_pred += batch_pred_edges

            base_comp_gid = 0
            for x in batch_data:
                identifier.append((x['pii'], x['t_idx']))
                comp_dict = dict()
                comp_dict['row'] = pred_comp_gid_labels[base_comp_gid:base_comp_gid+x['num_rows']]
                base_comp_gid += x['num_rows']
                comp_dict['col'] = pred_comp_gid_labels[base_comp_gid:base_comp_gid+x['num_cols']]
                base_comp_gid += x['num_cols']
                ret_comp_pred.append(comp_dict)

    comp_gid_metrics = pd.DataFrame(classification_report(y_comp_true, y_comp_pred, labels=[1, 2, 3], output_dict=True))
    prec, recall, fscore, _ = precision_recall_fscore_support(y_edge_true, y_edge_pred, average='binary')
    edge_metrics = {'precision': prec, 'recall': recall, 'fscore': fscore}

    ret_tuples_pred, all_pred_tuples, type_labels = [], [], []
    for (pii, t_idx), edges, comp_gid_pred in zip(identifier, ret_edge_pred, ret_comp_pred):
        pred_tuples, label = get_pred_tuples(pii, t_idx, edges, comp_gid_pred, split)
        ret_tuples_pred.append(pred_tuples)
        all_pred_tuples += pred_tuples
        type_labels.append(label)

    tuple_metrics = get_tuples_metrics(gold_tuples[split], all_pred_tuples)
    composition_metrics = get_composition_metrics(gold_tuples[split], all_pred_tuples)

    if not debug:
        return comp_gid_metrics.iloc[:3, :3], edge_metrics, tuple_metrics, composition_metrics
    else:
        return identifier, (comp_gid_metrics.iloc[:3, :3], y_comp_pred, ret_comp_pred), (edge_metrics, ret_edge_pred), \
        (tuple_metrics, ret_tuples_pred), composition_metrics, type_labels


for seed in args.seeds:
    set_seed(seed)

    for alpha in alphas:

        print(f'\nSeed {seed} Alpha {alpha}')

        model_path = f"../../models/model_{args.model_variant}_{seed}.bin"

        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model = model.to(device)

        res = dict()
        for s in splits[1:]:
            res[s] = dict()
            res[s]['identifier'], (res[s]['comp_gid_stats'], res[s]['comp_gid_pred_orig'], res[s]['comp_gid_pred']), \
            (res[s]['edge_stats'], res[s]['edge_pred']), (res[s]['tuple_metrics'], res[s]['tuples_pred']), \
            res[s]['composition_metrics'], res[s]['type_labels'] = eval_model(s, alpha, debug=True)

            if s not in ['val', 'test']:
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
                if s not in ['val', 'test']:
                    print(f'{s} {n} violations: {violations}/{total}')
                res[s][n] = violations

        os.makedirs(os.path.join(table_dir, 'res_dir'), exist_ok=True)
        pickle.dump(res, open(os.path.join(table_dir, 'res_dir', f'res_{args.model_variant}_mid_alpha_{alpha}_{seed}.pkl'), 'wb'))
        # os.remove(args.model_save_file)
