from argparse import ArgumentParser
import math
import multiprocessing as mp
import os
import pickle
import sys
sys.path.append('..')

from sklearn.metrics import precision_recall_fscore_support
from sklearn.utils.class_weight import compute_class_weight
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import set_seed, get_linear_schedule_with_warmup

from gnn_1_model import GNN_1_Model as Model
from regex_lib import parse_composition
from utils import *


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print('using device:', device)


parser = ArgumentParser()
parser.add_argument('--seeds', nargs='+', required=True, type=int)
parser.add_argument('--hidden_layer_sizes', nargs='+', required=True, type=int)
parser.add_argument('--num_heads', nargs='+', required=True, type=int)
parser.add_argument('--use_regex_feat', action='store_true')
parser.add_argument('--use_max_freq_feat', action='store_true')
parser.add_argument('--add_constraint', action='store_true')
parser.add_argument('--regex_emb_size', required=False, default=256, type=int)
parser.add_argument('--max_freq_emb_size', required=False, default=256, type=int)
parser.add_argument('--model_variant', required=True, type=str)
args = parser.parse_args()
print(args)

alphas = [0.4, 0.5, 0.6, 0.7, 0.8]

lm_name = 'm3rg-iitd/matscibert'
cache_dir = os.path.join(table_dir, '.cache')

if args.use_regex_feat:
    for c in tqdm(comp_data):
        c['regex_feats'] = get_regex_feats(c['act_table'])

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

all_train_regex_labels = [x['regex_table'] for x in datasets['train']]
all_train_gid_labels = []
for x in datasets['train']:
    all_train_gid_labels += x['gid_row_label'] + x['gid_col_label']

model_args = {
    'hidden_layer_sizes': args.hidden_layer_sizes,
    'num_heads': args.num_heads,
    'lm_name': lm_name,
    'cache_dir': cache_dir,
    'use_regex_feat': args.use_regex_feat,
    'use_max_freq_feat': args.use_max_freq_feat,
    'add_constraint': args.add_constraint,
    'regex_emb_size': args.regex_emb_size,
    'max_freq_emb_size': args.max_freq_emb_size,
}
model = Model(model_args).to(device)

gold_tuples = dict()
for split in ['val', 'test']:
    gold_tuples[split] = []
    for pii, t_idx in train_val_test_split[split]:
        gold_tuples[split] += get_gold_tuples(pii, t_idx)


def get_pred_tuples(pii_t_idx: tuple, regex_table: list, orient: str, gid):
    gid_list = []
    c = comp_data_dict[pii_t_idx]
    table = c['act_table']
    pii, t_idx = pii_t_idx
    if orient == 'row':
        for i in range(len(table)):
            if gid is not None and table[i][gid]:
                gid_list.append('_' + table[i][gid])
            else:
                gid_list.append('')
    else:
        for j in range(len(table[0])):
            if gid is not None and table[gid][j]:
                gid_list.append('_' + table[gid][j])
            else:
                gid_list.append('')
    tuples = []
    for i in range(len(table)):
        for j in range(len(table[0])):
            if regex_table[i][j] is None: continue
            prefix = f'{pii}_{t_idx}_{i}_{j}_0'
            for x in regex_table[i][j]:
                if x[1] == 0: continue
                gid = gid_list[i] if orient == 'row' else gid_list[j]
                tuples.append((prefix + gid, x[0], x[1], pred_cell_mol_wt(c, i, j)))
    return tuples


def get_regex_table_and_orient(table):
    regex_table = []
    regex_label = 0
    for r in table:
        res_r = []
        for cell in r:
            comp = parse_composition(cell)
            if len(comp) == 0 or len(comp[0][0]) == 1:
                res_r.append(None)
                continue
            l = comp[0][0]
            new_l = []
            for x in l:
                if type(x[1]) == float:
                    x = (x[0], round(x[1], 5))
                elif type(x[1]) == int:
                    x = (x[0], float(x[1]))
                new_l.append(x)
            if all(type(x[1]) == float for x in new_l):
                regex_label = 1
                res_r.append(new_l)
            else:
                res_r.append(None)
        regex_table.append(res_r)
    if regex_label == 0:
        return None, None
    row_max = 0
    for r in range(len(table)):
        curr = 0
        for comp in regex_table[r]:
            if type(comp) == list:
                curr += 1
        row_max = max(row_max, curr)
    col_max = 0
    for c in range(len(table[0])):
        curr = 0
        for r in range(len(table)):
            if type(regex_table[r][c]) == list:
                curr += 1
        col_max = max(col_max, curr)
    if row_max <= col_max:
        return regex_table, 'row'
    return regex_table, 'col'


def get_gid_labels_and_tuples(gid_logits, scc_label: int, pii_t_idx: tuple, num_rows: int, num_cols: int, mid_alpha):
    row_gid_labels, col_gid_labels = [0] * num_rows, [0] * num_cols
    if scc_label == 0:
        return row_gid_labels + col_gid_labels, []
    regex_table, orient = get_regex_table_and_orient(comp_data_dict[pii_t_idx]['act_table'])
    if orient is None:
        return row_gid_labels + col_gid_labels, []
    gid = None
    if orient == 'row':
        gid_col_probs = F.softmax(gid_logits[num_rows:], dim=1)
        gid_idx = gid_col_probs[:, 1].argmax()
        if gid_col_probs[gid_idx, 1] > mid_alpha:
            col_gid_labels[gid_idx] = 1
            gid = gid_idx
    else:
        gid_row_probs = F.softmax(gid_logits[:num_rows], dim=1)
        gid_idx = gid_row_probs[:, 1].argmax()
        if gid_row_probs[gid_idx, 1] > mid_alpha:
            row_gid_labels[gid_idx] = 1
            gid = gid_idx
    return row_gid_labels + col_gid_labels, get_pred_tuples(pii_t_idx, regex_table, orient, gid)


def get_batch_gid_labels_and_tuples(gid_logits, scc_labels: list, pii_t_idxs: list, num_rows: list, num_cols: list, mid_alpha):
    base_gid = 0
    pred_gid_labels, pred_tuples = [], []
    for pii_t_idx, regex_label, r, c in zip(pii_t_idxs, scc_labels, num_rows, num_cols):
        num_gid_logits = r + c
        gids_labels, tuples = get_gid_labels_and_tuples(
            gid_logits[base_gid:base_gid+num_gid_logits], regex_label, pii_t_idx, r, c, mid_alpha)
        pred_gid_labels += gids_labels
        pred_tuples.append(tuples)
        base_gid += num_gid_logits
    return pred_gid_labels, pred_tuples


def eval_model(split, mid_alpha, debug=False):
    model.eval()
    identifier = []
    y_scc_true, y_scc_pred = [], []
    y_gids_true, y_gids_pred, ret_gids_pred = [], [], []
    ret_tuples_pred = []
    y_true_scc_gids, ret_true_scc_gids, ret_true_scc_tuples = [], [], []

    with torch.no_grad():
        tepoch = tqdm(loaders[split], unit='batch')
        for batch_data in tepoch:
            tepoch.set_description(f'{split} mode')
            (scc_logits, scc_labels), (gid_logits, gid_labels) = model(batch_data)
            true_regex_labels = scc_labels.cpu().detach().tolist()
            pred_regex_labels = scc_logits.argmax(1).cpu().detach().tolist()
            y_scc_true += true_regex_labels
            y_scc_pred += pred_regex_labels
            
            y_gids_true += gid_labels.cpu().detach().tolist()
            gid_probs = F.softmax(gid_logits, dim=1)
            pred_gid_labels = gid_logits.argmax(1).cpu().detach().tolist()
            pred_gid_labels_1 = gid_probs.argmax(1).cpu().detach().tolist()
            assert pred_gid_labels == pred_gid_labels_1
            
            for i, (pred_prob, pred_label) in enumerate(zip(gid_probs, pred_gid_labels)):
                # if labelled gid but logit value after softmax is less than mid_alpha then label 0
                if pred_label == 1:
                    assert gid_probs[i, 1] == pred_prob[1]
                    assert gid_logits[i, 1] >= gid_logits[i, 0]
                    if pred_prob[1] < mid_alpha:
                        pred_gid_labels[i] = 0

            base = 0
            for p, x in zip(pred_regex_labels, batch_data):
                if p == 1:
                    y_gids_pred += pred_gid_labels[base:base+x['num_rows']+x['num_cols']]
                else:
                    y_gids_pred += [0] * (x['num_rows'] + x['num_cols'])
                base += x['num_rows'] + x['num_cols']

            if debug:
                base = 0
                for p, x in zip(true_regex_labels, batch_data):
                    if p == 1:
                        y_true_scc_gids += pred_gid_labels[base:base+x['num_rows']+x['num_cols']]
                    else:
                        y_true_scc_gids += [0] * (x['num_rows'] + x['num_cols'])
                    base += x['num_rows'] + x['num_cols']

            num_rows, num_cols = [x['num_rows'] for x in batch_data], [x['num_cols'] for x in batch_data]
            pii_t_idxs = [(x['pii'], x['t_idx']) for x in batch_data]
            identifier += pii_t_idxs
            pred_gid_labels, pred_tuples = get_batch_gid_labels_and_tuples(
                gid_logits.cpu().detach(), pred_regex_labels, pii_t_idxs, num_rows, num_cols, mid_alpha)
            ret_tuples_pred += pred_tuples

            if not debug: continue
            base_gid = 0
            for x in batch_data:
                gid_dict = dict()
                gid_dict['row'] = pred_gid_labels[base_gid:base_gid+x['num_rows']]
                base_gid += x['num_rows']
                gid_dict['col'] = pred_gid_labels[base_gid:base_gid+x['num_cols']]
                base_gid += x['num_cols']
                ret_gids_pred.append(gid_dict)
            
            pred_gid_labels, pred_tuples = get_batch_gid_labels_and_tuples(
                gid_logits.cpu().detach(), true_regex_labels, pii_t_idxs, num_rows, num_cols, mid_alpha)
            ret_true_scc_tuples += pred_tuples
            for x in batch_data:
                gid_dict = dict()
                gid_dict['row'] = pred_gid_labels[base_gid:base_gid+x['num_rows']]
                base_gid += x['num_rows']
                gid_dict['col'] = pred_gid_labels[base_gid:base_gid+x['num_cols']]
                base_gid += x['num_cols']
                ret_true_scc_gids.append(gid_dict)

    prec, recall, fscore, _ = precision_recall_fscore_support(y_scc_true, y_scc_pred, average='binary')
    scc_metrics = {'precision': prec, 'recall': recall, 'fscore': fscore}
    prec, recall, fscore, _ = precision_recall_fscore_support(y_gids_true, y_gids_pred, average='binary')
    gids_metrics = {'precision': prec, 'recall': recall, 'fscore': fscore}

    all_pred_tuples = []
    for t in ret_tuples_pred:
        all_pred_tuples += t
    tuple_metrics = get_tuples_metrics(gold_tuples[split], all_pred_tuples)
    composition_metrics = get_composition_metrics(gold_tuples[split], all_pred_tuples)

    if not debug:
        return scc_metrics, gids_metrics, tuple_metrics, composition_metrics
    else:
        return identifier, (scc_metrics, y_scc_pred), (gids_metrics, y_gids_pred, ret_gids_pred), \
        (y_true_scc_gids, ret_true_scc_gids), (tuple_metrics, ret_tuples_pred), \
        (ret_true_scc_tuples, ), composition_metrics


for seed in args.seeds:
    set_seed(seed)

    for alpha in alphas:

        print(f'\nSeed {seed} Alpha {alpha}')

        model_path = f"../../models/model_{args.model_variant}_{seed}.bin"

        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model = model.to(device)

        res = {'val': dict(), 'test': dict()}
        print()
        for s in res.keys():
            res[s]['identifier'], (res[s]['scc_stats'], res[s]['scc_pred']), (res[s]['gid_stats'], \
            res[s]['gid_pred_orig'], res[s]['gid_pred']), (res[s]['true_scc_gid_pred_orig'], \
            res[s]['true_scc_gid_pred']), (res[s]['tuple_metrics'], res[s]['tuples_pred']), \
            (res[s]['true_scc_tuples_pred'], ), res[s]['composition_metrics'] = eval_model(s, alpha, debug=True)
            print(f'{s} scc: \n', res[s]['scc_stats'])
            print(f'{s} gid: \n', res[s]['gid_stats'])
            print(f'{s} tuple metrics: \n', res[s]['tuple_metrics'])
            print(f'{s} composition metrics: \n', res[s]['composition_metrics'])

            for k in ['gid_pred_orig', 'true_scc_gid_pred_orig']:
                gid_pred_orig = []
                base = 0
                for pii_t_idx in res[s]['identifier']:
                    c = comp_data_dict[pii_t_idx]
                    d = dict()
                    d['row'] = res[s][k][base:base+c['num_rows']]
                    base += c['num_rows']
                    d['col'] = res[s][k][base:base+c['num_cols']]
                    base += c['num_cols']
                    gid_pred_orig.append(d)
                res[s][k] = gid_pred_orig

            violations, total = 0, 0
            for table in res[s]['gid_pred_orig']:
                v, t = cnt_3_3_violations(table)
                violations += v
                total += t
            print(f'{s} 3_3_violations: {violations}/{total}')
            res[s]['3_3_violations'] = violations

        os.makedirs(os.path.join(table_dir, 'res_dir'), exist_ok=True)
        pickle.dump(res, open(os.path.join(table_dir, 'res_dir', f'res_{args.model_variant}_mid_alpha_{alpha}_{seed}.pkl'), 'wb'))
        # os.remove(args.model_save_file)
