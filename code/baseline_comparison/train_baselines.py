from argparse import ArgumentParser
import math
import multiprocessing as mp
import os
import pickle

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

from utils import *


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print('using device:', device)


parser = ArgumentParser()
parser.add_argument('--seed', required=True, type=int)
parser.add_argument('--model', choices=['gat', 'tapas', 'tapas_adapted'], required=True, type=str)
parser.add_argument('--hidden_layer_sizes', nargs='+', type=int)
parser.add_argument('--num_heads', nargs='+', type=int)
parser.add_argument('--num_epochs', required=False, default=30, type=int)
parser.add_argument('--lr', required=False, default=1e-3, type=float)
parser.add_argument('--lm_lr', required=False, default=1e-5, type=float)
parser.add_argument('--e_loss_lambda', required=False, default=3.0, type=float)
parser.add_argument('--model_save_file', required=True, type=str)
parser.add_argument('--res_file', required=False, type=str)
args = parser.parse_args()
print(args)

if args.model == 'tapas':
    from tapas_baseline_model import TapasBaselineModel as Model
elif args.model == 'gat':
    if args.hidden_layer_sizes is None or args.num_heads is None:
        print('Hidden layer sizes or Num heads not provided')
        exit(1)
    from gat_model import GATModel as Model
elif args.model == 'tapas_adapted':
    from tapas_adapted_baseline_model import TapasAdaptedBaselineModel as Model
else:
    raise NotImplementedError

cache_dir = os.path.join(table_dir, '.cache')
os.makedirs(os.path.dirname(os.path.abspath(args.model_save_file)), exist_ok=True)

if args.model == 'tapas':
    lm_name = 'google/tapas-base'
else:
    lm_name = 'm3rg-iitd/matscibert'

torch.set_deterministic(True)
torch.backends.cudnn.benchmark = False

datasets = dict()
for split in splits:
    datasets[split] = TableDataset([comp_data_dict[pii_t_idx] for pii_t_idx in train_val_test_split[split]])

set_seed(args.seed)
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
for x in tqdm(datasets['train']):
    table_edges = Model.get_edges(x['num_rows'], x['num_cols'])
    all_train_edge_labels += Model.create_edge_labels(table_edges, x['edge_list']).tolist()

num_epochs = args.num_epochs
n_batches = math.ceil(len(datasets['train']) / batch_size)
n_steps = n_batches * num_epochs
warmup_steps = n_steps // 10

model_args = {
    'hidden_layer_sizes': args.hidden_layer_sizes,
    'num_heads': args.num_heads,
    'lm_name': lm_name,
    'cache_dir': cache_dir,
}
model = Model(model_args).to(device)

optim_grouped_parameters = [
    {'params': [p for n, p in model.named_parameters() if 'encoder' not in n], 'lr': args.lr},
    {'params': [p for n, p in model.named_parameters() if 'encoder' in n], 'lr': args.lm_lr},
]
optim = torch.optim.AdamW(optim_grouped_parameters)

comp_gid_class_weights = Tensor(compute_class_weight('balanced', classes=[0, 1, 2, 3], y=all_train_comp_labels)).to(device)
comp_gid_loss_fn = nn.CrossEntropyLoss(weight=comp_gid_class_weights)

neg_edge_wt, pos_edge_wt = compute_class_weight('balanced', classes=[0, 1], y=all_train_edge_labels)
edge_loss_fn = nn.BCEWithLogitsLoss(reduction='none')

scheduler = get_linear_schedule_with_warmup(optim, num_warmup_steps=warmup_steps, num_training_steps=n_steps)

gold_tuples = dict()
for split in ['val', 'test']:
    gold_tuples[split] = []
    for pii, t_idx in train_val_test_split[split]:
        gold_tuples[split] += get_gold_tuples(pii, t_idx)


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

    df = pd.DataFrame(Model.get_edges(num_rows, num_cols).tolist())
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


losses = ['row_col_gid', 'edge']
coeffs = [1.0, args.e_loss_lambda]


def train_model(epoch):
    model.train()
    epoch_loss = {l: 0.0 for l in losses}

    n_batches = len(loaders['train'])
    tepoch = tqdm(loaders['train'], unit='batch')
    batch_loss = dict()

    for batch_data in tepoch:
        tepoch.set_description(f'Epoch {epoch}')
        torch.cuda.empty_cache()
        (comp_gid_logits, comp_gid_labels), (edge_logits, edge_labels) = model(batch_data)
        batch_loss[losses[0]] = comp_gid_loss_fn(comp_gid_logits, comp_gid_labels)
        edge_loss_vec = edge_loss_fn(edge_logits, edge_labels.to(dtype=torch.float32))
        class_wt = torch.ones(len(edge_loss_vec)) * neg_edge_wt
        class_wt[edge_labels == 1] = pos_edge_wt
        batch_loss[losses[1]] = (edge_loss_vec * class_wt.to(device)).mean()
        for l in losses:
            epoch_loss[l] += batch_loss[l].item()
        loss = sum(coeffs[i] * batch_loss[losses[i]] for i in range(len(losses)))
        optim.zero_grad()
        loss.backward()
        optim.step()
        scheduler.step()
        del comp_gid_logits, comp_gid_labels, edge_logits, edge_labels, edge_loss_vec, class_wt

    for l in losses:
        epoch_loss[l] /= n_batches
    return epoch_loss


def eval_model(split, debug=False):
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
            y_comp_pred += comp_gid_logits.argmax(1).cpu().detach().tolist()
            num_rows, num_cols = [x['num_rows'] for x in batch_data], [x['num_cols'] for x in batch_data]
            pred_comp_gid_labels, pred_edge_labels, batch_pred_edges = get_batch_edge_gid_pred_labels(
                comp_gid_logits.cpu().detach(), edge_logits.cpu().detach(), num_rows, num_cols)
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

    ret_tuples_pred, all_tuples_pred = [], []
    for (pii, t_idx), edges, comp_gid_pred in zip(identifier, ret_edge_pred, ret_comp_pred):
        tuples_pred = get_sum_100_pred_tuples(pii, t_idx, edges, comp_gid_pred)
        ret_tuples_pred.append(tuples_pred)
        all_tuples_pred += tuples_pred

    tuple_metrics = get_tuples_metrics(gold_tuples[split], all_tuples_pred)
    composition_metrics = get_composition_metrics(gold_tuples[split], all_tuples_pred)

    if not debug:
        return comp_gid_metrics.iloc[:3, :3], edge_metrics, tuple_metrics, composition_metrics
    else:
        return identifier, (comp_gid_metrics.iloc[:3, :3], y_comp_pred, ret_comp_pred), (edge_metrics, ret_edge_pred), \
        (tuple_metrics, ret_tuples_pred), composition_metrics


best_val = 0.0

for epoch in range(num_epochs):
    epoch_loss = train_model(epoch)
    print(f'Epoch {epoch} | Loss {epoch_loss}')
    val_stats = eval_model('val')
    print('Val Stats\n', val_stats)
    test_stats = eval_model('test')
    print('Test Stats\n', test_stats)
    print()

    if val_stats[-1]['fscore'] > best_val:
        best_val = val_stats[-1]['fscore']
        torch.save(model.state_dict(), args.model_save_file)


model.load_state_dict(torch.load(args.model_save_file, map_location=torch.device('cpu')))
model = model.to(device)

res = dict()
for s in splits[1:]:
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

if args.res_file:
    os.makedirs(os.path.join(table_dir, 'res_dir'), exist_ok=True)
    pickle.dump(res, open(os.path.join(table_dir, 'res_dir', args.res_file), 'wb'))
    # os.remove(args.model_save_file)
