from collections import Counter, defaultdict
import os
import pickle
import re
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sympy import sympify, solve
from torch.utils.data import Dataset

sys.path.append('..')
from regex_lib import *


table_dir = '../../data'
comp_data = pickle.load(open(os.path.join(table_dir, 'train_data.pkl'), 'rb')) + \
            pickle.load(open(os.path.join(table_dir, 'val_test_data.pkl'), 'rb'))

for c in comp_data:
    if c['regex_table'] == 1:
        c['row_label'] = [0] * c['num_rows']
        c['col_label'] = [0] * c['num_cols']
        c['edge_list'] = []
    if sum(c['gid_row_label']) == 1:
        c['row_label'][c['gid_row_label'].index(1)] = 3
    elif sum(c['gid_col_label']) == 1:
        c['col_label'][c['gid_col_label'].index(1)] = 3

train_val_test_split = pickle.load(open(os.path.join(table_dir, 'train_val_test_split.pkl'), 'rb'))
comp_data_dict = {(c['pii'], c['t_idx']): c for c in comp_data}

splits = [
    'train',
    'val', 'val_non_scc', 'val_mcc_ci', 'val_mcc_pi',
    'test', 'test_non_scc', 'test_mcc_ci', 'test_mcc_pi',
]

for split in ['train', 'val', 'test']:
    if split == 'train':
        train_val_test_split[split] = [x for x in train_val_test_split[split] if comp_data_dict[x]['regex_table'] == 0]
    else:
        train_val_test_split[f'{split}_non_scc'] = [x for x in train_val_test_split[split] if comp_data_dict[x]['regex_table'] == 0]
        train_val_test_split[f'{split}_mcc_ci'] = [x for x in train_val_test_split[f'{split}_non_scc'] if comp_data_dict[x]['sum_less_100'] == 0 and comp_data_dict[x]['comp_table']]
        train_val_test_split[f'{split}_mcc_pi'] = [x for x in train_val_test_split[f'{split}_non_scc'] if comp_data_dict[x]['sum_less_100'] == 1]


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


elements_compounds = pickle.load(open(os.path.join(table_dir, 'elements_compounds.pkl'), 'rb'))
comp_num_pattern = r'(\d+\.\d+|\d+/\d+|\d+)'
ele_num = r'((' + '|'.join(elements_compounds['elements']) + r')' + comp_num_pattern + r'?)'
many_ele_num = r'(' + ele_num + r')+'
ele_comp_pattern = r'(((\(' + many_ele_num + '\)' + comp_num_pattern + r')|(' + many_ele_num + r'))+)'
ele_comp_pattern = r'(?:^|\W)(' + ele_comp_pattern + r'|Others|Other)(?:\W|$)'
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


def get_mcc_ci_pred_tuples(pii, t_idx, edges, comp_gid_pred):
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


extracted_regex = pickle.load(open(os.path.join(table_dir, 'extracted_regex.pkl'), 'rb'))
all_elements, all_compounds = elements_compounds['elements'], elements_compounds['compounds']
comp_pattern = re.compile(r'(?:^|[^a-zA-Z])(' + '|'.join(all_compounds + all_elements) + r')(?:[^a-zA-Z]|$)')
comp_pattern_1 = re.compile(r'(' + '|'.join(all_compounds+all_elements) + r')')
var_pattern = re.compile(r'(?:^|[^\w-])(' + '|'.join(comp_vars) + r')')
var_pattern_1 = re.compile(r'(' + '|'.join(comp_vars) + r')')


def get_comp_vars_and_nums(table):
    comps, vars, nums = [], [], []
    for r in table:
        r_comps, r_vars, r_nums = [], [], []
        for cell in r:
            found_compounds = re.findall(comp_pattern, cell)
            r_comps.append(found_compounds)
            subs_cell = re.sub(comp_pattern_1, ' ', cell)
            found_vars = list(set(m.group(1) for m in re.finditer(var_pattern, subs_cell)))
            r_vars.append(found_vars)
            subs_cell = re.sub(var_pattern_1, ' ', subs_cell)
            cell_nums = re.findall(num, subs_cell)
            cell_nums = list(map(float, cell_nums))
            r_nums.append(cell_nums[0] if len(cell_nums) > 0 else None)
        comps.append(r_comps)
        vars.append(r_vars)
        nums.append(r_nums)
    return comps, vars, nums


def get_clf_feats(c, row_labels, col_labels, edges):
    comps, vars, nums = get_comp_vars_and_nums(c['act_table'])
    edge_dict = {src: dst for src, dst in edges}
    dsts = sorted(set([e[1] for e in edges]))
    found_vars, found_compounds = [], []
    for dst in dsts:
        found_compounds += comps[dst[0]][dst[1]]
        found_vars += vars[dst[0]][dst[1]]
    found_compounds, found_vars = set(found_compounds), set(found_vars)
    max_s, avg_s = 0, 0
    if 1 in row_labels:
        comp_rows = [r for r in range(c['num_rows']) if row_labels[r] == 1]
        comp_cols = [j for j in range(c['num_cols']) if col_labels[j] == 2]
        for r in comp_rows:
            s = 0
            for j in comp_cols:
                if nums[r][j]:
                    s += nums[r][j]
            max_s = max(max_s, s)
            avg_s += s
        avg_s /= len(comp_rows)
    else:
        comp_rows = [r for r in range(c['num_rows']) if row_labels[r] == 2]
        comp_cols = [j for j in range(c['num_cols']) if col_labels[j] == 1]
        for j in comp_cols:
            s = 0
            for r in comp_rows:
                if nums[r][j]:
                    s += nums[r][j]
            max_s = max(max_s, s)
            avg_s += s
        avg_s /= len(comp_cols)
    return [len(found_compounds), len(found_vars), avg_s, max_s, (row_labels + col_labels).count(2)]


def get_mcc_clf():
    X, y = [], []
    for pii_t_idx in train_val_test_split['train']:
        c = comp_data_dict[pii_t_idx]
        if not c['comp_table']: continue
        edges = []
        for e in c['edge_list']:
            edges.append([
                (e[0] // c['num_cols'], e[0] % c['num_cols']),
                (e[1] // c['num_cols'], e[1] % c['num_cols'])
            ])
        X.append(get_clf_feats(c, c['row_label'], c['col_label'], edges))
        y.append(c['sum_less_100'])
    clf = LogisticRegression(random_state=0).fit(X, y)
    return clf


clf = get_mcc_clf()


def get_mcc_pi_pred_tuples(pii, t_idx, edges, comp_gid_pred):
    cc = comp_data_dict[(pii, t_idx)]
    row_labels, col_labels = comp_gid_pred['row'], comp_gid_pred['col']
    tuples = []
    if sum(row_labels) == 0 or sum(col_labels) == 0: return tuples
    comps, vars, nums = get_comp_vars_and_nums(cc['act_table'])
    for i in range(cc['num_rows']):
        for j in range(cc['num_cols']):
            comps[i][j] += vars[i][j]
    del vars
    edge_dict = {src: dst for src, dst in edges}

    keys_left = set(k for k in extracted_regex[pii].keys() if type(k) == str) - set(['Title', 'Abstract'])
    keys_left = sorted([k for k in keys_left if not k.startswith('Appendix') and not k.endswith('_footer')], key=lambda x: int(x.split('_')[0]))
    regex_comps = []
    for k in [t_idx, f'{t_idx}_footer', 'Title', 'Abstract'] + keys_left:
        regex_comps += extracted_regex[pii][k]

    all_nums = []
    for r in range(cc['num_rows']):
        for c in range(cc['num_cols']):
            if row_labels[r] * col_labels[c] == 2 and nums[r][c] is not None:
                cx, cy = edge_dict[(r, c)]
                if len(comps[cx][cy]) > 0:
                    all_nums.append(nums[r][c])
    if any(n > 1 for n in all_nums):
        ds = [100]
    else:
        ds = [1, 100]

    if 1 in row_labels:
        comp_cols = [j for j in range(cc['num_cols']) if col_labels[j] == 2]
        gid_cols = [j for j in range(cc['num_cols']) if col_labels[j] == 3]
        gid_col = gid_cols[0] if gid_cols else None
        for r in range(cc['num_rows']):
            if row_labels[r] != 1: continue
            vars, compounds = dict(), dict()
            first_comp_col = -1
            for c in comp_cols:
                if nums[r][c] is None: continue
                cx, cy = edge_dict[(r, c)]
                if len(comps[cx][cy]) == 0: continue
                if first_comp_col == -1: first_comp_col = c
                if comps[cx][cy][0] in comp_vars:
                    if comps[cx][cy][0].lower() not in vars:
                        vars[comps[cx][cy][0].lower()] = nums[r][c]
                else:
                    if comps[cx][cy][0] not in compounds:
                        compounds[comps[cx][cy][0]] = nums[r][c]
            if len(vars) == 0 and len(compounds) == 0: continue
            assert first_comp_col != -1
            found = False
            if len(vars) > 0:
                for comp, _ in regex_comps:
                    assert isinstance(comp, list)
                    this_comp_vars = set()
                    for _, p in comp:
                        if type(p) != str: continue
                        for v in comp_vars:
                            if v in p:
                                this_comp_vars.add(v.lower())
                    if this_comp_vars != set(vars.keys()): continue
                    new_comp = []
                    for ele_comp, p in comp:
                        if type(p) != str:
                            new_comp.append((ele_comp, p))
                        else:
                            p = p.lower()
                            for var, val in vars.items():
                                p = p.replace(var, str(val))
                            new_comp.append((ele_comp, eval_expr(p)))
                    if all(0 <= p <= 1 for _, p in new_comp):
                        found = True
                        break
            else:
                for comp, _ in regex_comps:
                    assert isinstance(comp, list)
                    this_compounds = [x[0] for x in comp]
                    if len(set(compounds.keys()) - set(this_compounds)) > 0: continue
                    comp_dict = {x[0]: x[1] for x in comp}
                    this_comp_vars = set()
                    for _, p in comp:
                        if type(p) != str: continue
                        for v in comp_vars:
                            if v in p:
                                this_comp_vars.add(v.lower())
                    for d in ds:
                        var_mapping = dict()
                        suitable_comp = True
                        for compound, p in compounds.items():
                            if p / d > 1:
                                suitable_comp = False
                                break
                            sol = solve(sympify(f'Eq({comp_dict[compound]}, {p/d})'))
                            if len(sol) == 0:
                                suitable_comp = False
                                break
                            if type(sol[0]) == dict or float(sol[0]) < 0:
                                suitable_comp = False
                                break
                            m = re.search(var_pattern_1, comp_dict[compound])
                            assert m is not None
                            var = m.group().lower()
                            if var in var_mapping and var_mapping[var] != float(sol[0]):
                                suitable_comp = False
                                break
                            var_mapping[var] = float(sol[0])

                        if not suitable_comp or set(var_mapping.keys()) != this_comp_vars: continue
                        new_comp = []
                        for ele_comp, p in comp:
                            if type(p) != str:
                                new_comp.append((ele_comp, p))
                            else:
                                p = p.lower()
                                for var, val in var_mapping.items():
                                    p = p.replace(var, str(val))
                                new_comp.append((ele_comp, eval_expr(p)))
                        if all(0 <= p <= 1 for _, p in new_comp):
                            found = True
                            break
                    if found: break
            if found:
                g_id = f'{pii}_{t_idx}_{r}_{first_comp_col}_0'
                if gid_col is not None and cc['act_table'][r][gid_col]:
                    g_id += '_' + cc['act_table'][r][gid_col]
                for ele_comp, p in new_comp:
                    if p > 0:
                        tuples.append((g_id, ele_comp, round(p, 5), pred_cell_mol_wt(cc, r, first_comp_col)))
    else:
        assert 1 in col_labels
        comp_rows = [i for i in range(cc['num_rows']) if row_labels[i] == 2]
        gid_rows = [i for i in range(cc['num_rows']) if row_labels[i] == 3]
        gid_row = gid_rows[0] if gid_rows else None
        for c in range(cc['num_cols']):
            if col_labels[c] != 1: continue
            vars, compounds = dict(), dict()
            first_comp_row = -1
            for r in comp_rows:
                if nums[r][c] is None: continue
                cx, cy = edge_dict[(r, c)]
                if len(comps[cx][cy]) == 0: continue
                if first_comp_row == -1: first_comp_row = r
                if comps[cx][cy][0] in comp_vars:
                    if comps[cx][cy][0].lower() not in vars:
                        vars[comps[cx][cy][0].lower()] = nums[r][c]
                else:
                    if comps[cx][cy][0] not in compounds:
                        compounds[comps[cx][cy][0]] = nums[r][c]
            
            if len(vars) == 0 and len(compounds) == 0: continue
            assert first_comp_row != -1
            found = False
            if len(vars) > 0:
                for comp, _ in regex_comps:
                    assert isinstance(comp, list)
                    this_comp_vars = set()
                    for _, p in comp:
                        if type(p) != str: continue
                        for v in comp_vars:
                            if v in p:
                                this_comp_vars.add(v.lower())
                    if this_comp_vars != set(vars.keys()): continue
                    new_comp = []
                    for ele_comp, p in comp:
                        if type(p) != str:
                            new_comp.append((ele_comp, p))
                        else:
                            p = p.lower()
                            for var, val in vars.items():
                                p = p.replace(var, str(val))
                            new_comp.append((ele_comp, eval_expr(p)))
                    if all(0 <= p <= 1 for _, p in new_comp):
                        found = True
                        break
            else:
                for comp, _ in regex_comps:
                    assert isinstance(comp, list)
                    this_compounds = [x[0] for x in comp]
                    if len(set(compounds.keys()) - set(this_compounds)) > 0: continue
                    comp_dict = {x[0]: x[1] for x in comp}
                    this_comp_vars = set()
                    for _, p in comp:
                        if type(p) != str: continue
                        for v in comp_vars:
                            if v in p:
                                this_comp_vars.add(v.lower())
                    for d in ds:
                        var_mapping = dict()
                        suitable_comp = True
                        for compound, p in compounds.items():
                            if p / d > 1:
                                suitable_comp = False
                                break
                            sol = solve(sympify(f'Eq({comp_dict[compound]}, {p/d})'))
                            if len(sol) == 0:
                                suitable_comp = False
                                break
                            if type(sol[0]) == dict or float(sol[0]) < 0:
                                suitable_comp = False
                                break
                            m = re.search(var_pattern_1, comp_dict[compound])
                            assert m is not None
                            var = m.group().lower()
                            if var in var_mapping and var_mapping[var] != float(sol[0]):
                                suitable_comp = False
                                break
                            var_mapping[var] = float(sol[0])

                        if not suitable_comp or set(var_mapping.keys()) != this_comp_vars: continue
                        new_comp = []
                        for ele_comp, p in comp:
                            if type(p) != str:
                                new_comp.append((ele_comp, p))
                            else:
                                p = p.lower()
                                for var, val in var_mapping.items():
                                    p = p.replace(var, str(val))
                                new_comp.append((ele_comp, eval_expr(p)))
                        if all(0 <= p <= 1 for _, p in new_comp):
                            found = True
                            break
                    if found: break
            if found:
                g_id = f'{pii}_{t_idx}_{first_comp_row}_{c}_0'
                if gid_row is not None and cc['act_table'][gid_row][c]:
                    g_id += '_' + cc['act_table'][gid_row][c]
                for ele_comp, p in new_comp:
                    if p > 0:
                        tuples.append((g_id, ele_comp, round(p, 5), pred_cell_mol_wt(cc, first_comp_row, c)))
    return tuples


def get_pred_tuples(pii, t_idx, edges, comp_gid_pred, split):
    c = comp_data_dict[(pii, t_idx)]

    def proc_edges(e):
        return (e[0][0] * c['num_cols'] + e[0][1], e[1][0] * c['num_cols'] + e[1][1])

    if split.endswith('mcc_ci'):
        return get_mcc_ci_pred_tuples(pii, t_idx, edges, comp_gid_pred), 1
    if split.endswith('mcc_pi'):
        return get_mcc_pi_pred_tuples(pii, t_idx, edges, comp_gid_pred), 2
    if sum(comp_gid_pred['row']) == 0 or sum(comp_gid_pred['col']) == 0:
        return [], 3 # NC

    y_pred = clf.predict([get_clf_feats(c, comp_gid_pred['row'], comp_gid_pred['col'], edges)])[0]
    if y_pred == 1:
        return get_mcc_pi_pred_tuples(pii, t_idx, edges, comp_gid_pred), 2 # MCC-PI
    else:
        return get_mcc_ci_pred_tuples(pii, t_idx, edges, comp_gid_pred), 1 # MCC-CI


def get_gold_tuples(pii, t_idx):
    c = comp_data_dict[(pii, t_idx)]
    tuples = []
    if 'full_comp' not in c or c['regex_table'] == 1:
        return tuples
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

