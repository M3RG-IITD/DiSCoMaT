"""
---------------------------------------------------------------------------------------------------------
SciGlass Database is obtained from : https://github.com/epam/SciGlass
We thank the repository owner for publically releasing the dataset. The license for the same is provided below.
---------------------------------------------------------------------------------------------------------
ODC Open Database License (ODbL)

Copyright (c) 2019 EPAM Systems

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
---------------------------------------------------------------------------------------------------------
"""

from collections import defaultdict, Counter
import os
import pickle
import re
import sys
sys.path.append('..')

from tqdm import tqdm
from sympy import solve, sympify

from regex_lib import *


table_dir = '../../data'

train_data = pickle.load(open(os.path.join(table_dir, 'train_data_mcc_ci.pkl'), 'rb'))
train_data = [c for c in train_data if c['regex_table'] == 0 and not c['comp_table']]
train_data_dict = {(c['pii'], c['t_idx']): c for c in train_data}
train_piis = list(set(pii for pii, _ in train_data_dict))

data = pickle.load(open(os.path.join(table_dir, 'train_val_test_paper_data.pkl'), 'rb'))
for pii in data.keys():
    data[pii]['tables_captions'] = ''

for c in train_data:
    data[c['pii']]['tables_captions'] += c['caption'].replace('\n', ' ') + '\n'

pii_glass_ids = pickle.load(open(os.path.join(table_dir, 'sciglass_pii_gids.pkl'), 'rb'))

def non_zero_cols(df):
    return df.T[df.astype(bool).sum(axis=0) > 0].index.to_list()

composition = {
    'mol': pickle.load(open(os.path.join(table_dir, 'sciglass_composition_mol.pkl'), 'rb')),
    'wt': pickle.load(open(os.path.join(table_dir, 'sciglass_composition_wt.pkl'), 'rb')),
}

gids = dict()
avail_glass_ids = set()
for k in composition.keys():
    composition[k] = composition[k][(composition[k].sum(axis=1).round() == 100)]
    gids[k] = set(composition[k].index) & set(pii_glass_ids['GLASNO'])
    composition[k] = composition[k].loc[gids[k]].sort_index()
    composition[k] = composition[k][non_zero_cols(composition[k])]
    avail_glass_ids |= gids[k]

extracted_regex = defaultdict(dict)
pii_constituents = dict()


def extract_regex_from_paper_text(pii):
    glass_ids = set(pii_glass_ids.loc[pii_glass_ids['PII'] == pii, 'GLASNO']) & avail_glass_ids
    if len(glass_ids) == 0: return
    constituents = set()
    for k in composition.keys():
        constituents |= set(non_zero_cols(composition[k].loc[glass_ids & gids[k]]))
    constituents -= set(['RO', 'RO2', 'R2O', 'R2O3'])
    pii_constituents[pii] = constituents

    if len(constituents) == 0: return

    for section, text in data[pii].items():
        extracted_regex[pii][section] = parse_composition(text, constituents)

for pii in tqdm(train_piis):
    extract_regex_from_paper_text(pii)

regex_piis = list(extracted_regex.keys())

for pii in regex_piis:
    remove = True
    for section in extracted_regex[pii].keys():
        extracted_regex[pii][section] = [c for c in extracted_regex[pii][section] if len(c[0]) > 1]
        if len(extracted_regex[pii][section]) > 0:
            remove = False
    if remove: extracted_regex.pop(pii)
regex_piis = list(extracted_regex.keys())

for pii in regex_piis:
    remove = True
    for l in extracted_regex[pii].values():
        for x in l:
            if type(x[0][0][1]) == str:
                remove = False
    if remove: extracted_regex.pop(pii)
regex_piis = list(extracted_regex.keys())

pii_vars = defaultdict(set)
for pii in regex_piis:
    l = []
    for ll in extracted_regex[pii].values():
        l += ll
    for c in l:
        assert type(c[0]) == list
        for x in c[0]:
            if type(x[1]) == str:
                for var in comp_vars:
                    if var in x[1]:
                        pii_vars[pii].add(var)

var_regex_piis = dict()
for pii in regex_piis:
    l = []
    for ll in extracted_regex[pii].values():
        l += ll
    l = [c[0] for c in l]
    var_regex_piis[pii] = []
    for c in l:
        var_comp = False
        for x in c:
            if type(x[1]) == str:
                var_comp = True
                break
        if not var_comp: continue
        match = False
        for cc, _ in var_regex_piis[pii]:
            if dict(c) == dict(cc):
                match = True
                break
        if match: continue
        vars = set()
        for x in c:
            if type(x[1]) == str:
                for var in comp_vars:
                    if var in x[1]:
                        vars.add(var)
        var_regex_piis[pii].append((c, vars))


def viewcomp(df):
    return df.T[df.astype(bool).sum(axis=0) > 0].T


num_pattern = re.compile(r'-?\d+\.\d+|-?\d+')

def get_cons_pattern(pii):
    comp_list = sorted(list(pii_constituents[pii]), key=lambda x: -len(x))
    comp_list = [c.replace('(', '\(').replace(')', '\)') for c in comp_list]
    return re.compile('|'.join(comp_list))

def get_var_pattern(pii):
    return re.compile(r'(?:^|[^\w-])(' + '|'.join(sorted(pii_vars[pii])) + r')')

def get_comp_and_nums(table, cons_pattern, var_pattern):
    comps, nums = [], []
    for r in table:
        r_comps, r_nums = [], []
        for cell in r:
            found_constituents = list(set(re.findall(cons_pattern, cell)))
            subs_cell = re.sub(cons_pattern, ' ', cell)
            found_vars = list(set(m.group(1) for m in re.finditer(var_pattern, subs_cell)))
            subs_cell = re.sub(var_pattern, ' ', subs_cell).lower()
            r_comps.append(found_constituents + found_vars)
            if found_constituents:
                cell_nums = re.findall(num_pattern, subs_cell)
                r_nums.append(list(map(float, cell_nums)))
            else:
                m = re.search(r'[a-z]', subs_cell)
                end_idx = m.start() if m is not None else len(subs_cell)
                cell_nums = re.findall(num_pattern, subs_cell[:end_idx])
                r_nums.append(list(map(float, cell_nums)))
        comps.append(r_comps)
        nums.append(r_nums)
    return comps, nums

tables_comp, tables_nums = dict(), dict()
for c in train_data:
    if c['pii'] not in regex_piis: continue
    k = (c['pii'], c['t_idx'])
    tables_comp[k], tables_nums[k] = get_comp_and_nums(c['act_table'], get_cons_pattern(c['pii']), get_var_pattern(c['pii']))


def match_num_in_table(pii_t_idx, num, regex_comp, regex_vars, db_comps, tol=1e-2):
    if num < 0: return -1
    regex_var = list(regex_vars)[0]
    subs_comp = dict()
    for comp, perc in regex_comp:
        try:
            subs_comp[comp] = eval_expr(perc.replace(regex_var, str(num))) if type(perc) == str else perc
        except ZeroDivisionError:
            return -1
    assert len(subs_comp) == len(regex_comp)
    if any(v < 0 for v in subs_comp.values()): return -1
    subs_comp = {k: v for k, v in subs_comp.items() if v > 0}
    
    for i in range(len(db_comps)):
        db_comp = dict(db_comps.iloc[i])
        db_comp = {k: v for k, v in db_comp.items() if v > 0}
        if set(subs_comp.keys()) != set(db_comp.keys()): continue
        match = True
        for k in db_comp.keys():
            if db_comp[k] * (1 - tol) <= subs_comp[k] <= db_comp[k] * (1 + tol): pass
            else:
                match = False
                break
        if match: return i
    return -1


def get_table_edges_for_regex_comp(pii_t_idx, regex_comp, regex_vars, db_comps):
    comps, nums = tables_comp[pii_t_idx], tables_nums[pii_t_idx]
    comp_locations = defaultdict(list)
    for i, r in enumerate(comps):
        for j, c in enumerate(r):
            for comp in c:
                comp_locations[comp].append((i, j))

    regex_var = list(regex_vars)[0]
    edges = []

    if regex_var in comp_locations:
        var_locs = comp_locations[regex_var]
        src = defaultdict(list)
        possible_num_locs = set()
        for x, y in var_locs:
            for i in range(x, len(comps)):
                possible_num_locs.add((i, y))
                src[(i, y)].append((x, y))
            for j in range(y, len(comps[0])):
                possible_num_locs.add((x, j))
                src[(x, j)].append((x, y))
        for x, y in possible_num_locs:
            if len(nums[x][y]) == 0: continue
            num = nums[x][y][0]
            if num < 0: continue
            if match_num_in_table(pii_t_idx, num, regex_comp, regex_vars, db_comps) != -1:
                for s in src[(x, y)]:
                    edges.append((s, (x, y)))
    
    for comp, perc in regex_comp:
        if type(perc) != str or comp not in comp_locations: continue
        var_locs = comp_locations[comp]
        src = defaultdict(list)
        possible_num_locs = set()
        for x, y in var_locs:
            for i in range(x, len(comps)):
                possible_num_locs.add((i, y))
                src[(i, y)].append((x, y))
            for j in range(y, len(comps[0])):
                possible_num_locs.add((x, j))
                src[(x, j)].append((x, y))
        for x, y in possible_num_locs:
            if len(nums[x][y]) == 0: continue
            num = nums[x][y][0]
            if num < 0: continue
            for d in [1, 100]:
                if num / d > 1: continue
                sol = solve(sympify(f'Eq({perc}, {num/d})'))
                if len(sol) == 0: continue
                assert len(sol) == 1
                if match_num_in_table(pii_t_idx, float(sol[0]), regex_comp, regex_vars, db_comps) != -1:
                    for s in src[(x, y)]:
                        edges.append((s, (x, y)))
                    break
    return edges


def l1_dist(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def get_farthest(p, l):
    res = l[0]
    for p_ in l:
        if l1_dist(p, p_) > l1_dist(p, res):
            res = p_
    return res

def get_table_edges(pii_t_idx):
    pii, t_idx = pii_t_idx
    glass_ids = set(pii_glass_ids.loc[pii_glass_ids['PII'] == pii, 'GLASNO']) & avail_glass_ids
    res = []
    if len(glass_ids) == 0: return res
    r, c = train_data_dict[pii_t_idx]['num_rows'], train_data_dict[pii_t_idx]['num_cols']
    for k in composition.keys():
        db_comps = viewcomp(composition[k].loc[glass_ids & gids[k]]) / 100
        for regex_comp, regex_vars in var_regex_piis[pii]:
            if len(regex_vars) == 1:
                res += get_table_edges_for_regex_comp(pii_t_idx, regex_comp, regex_vars, db_comps)
    edges = sorted(set(res))
    
    edges_src, edges_dst = defaultdict(set), defaultdict(set)
    for src, dst in edges:
        edges_src[src].add(dst)
    for src in edges_src.keys():
        if src in edges_src[src]:
            edges_src[src] = set([src])
        for dst in edges_src[src]:
            edges_dst[dst].add(src)
    for dst in edges_dst.keys():
        if dst in edges_dst[dst]:
            edges_dst[dst] = set([dst])
        edges_dst[dst] = list(edges_dst[dst])
    edges = []
    for dst in edges_dst.keys():
        if len(edges_dst[dst]) == 1:
            edges.append((dst, edges_dst[dst][0]))
            continue
        edges.append((dst, get_farthest(dst, edges_dst[dst])))

    return sorted(edges)

regex_edges = dict()
for c in tqdm(train_data):
    if c['pii'] not in regex_piis: continue
    k = (c['pii'], c['t_idx'])
    edges = get_table_edges(k)
    if len(edges) > 0:
        regex_edges[k] = edges

orient = dict()
for pii_t_idx in regex_edges.keys():
    if len(regex_edges[pii_t_idx]) == 1:
        e = regex_edges[pii_t_idx][0]
        if e[0] == e[1]: # self edge
            pass
        elif e[0][1] == e[1][1]: # same column
            orient[pii_t_idx] = 'row'
        elif e[0][0] == e[1][0]: # same row
            orient[pii_t_idx] = 'col'
    else:
        srcs = [e[0] for e in regex_edges[pii_t_idx]]
        rows, cols = set([x[0] for x in srcs]), set([x[1] for x in srcs])
        assert len(rows) > 1 or len(cols) > 1
        if len(cols) == 1:
            orient[pii_t_idx] = 'row'
        elif len(rows) == 1:
            orient[pii_t_idx] = 'col'
        else:
            if all(e[0] != e[1] for e in regex_edges[pii_t_idx]):
                same_cols = sum(e[0][1] == e[1][1] for e in regex_edges[pii_t_idx])
                same_rows = sum(e[0][0] == e[1][0] for e in regex_edges[pii_t_idx])
                orient[pii_t_idx] = 'row' if same_cols >= same_rows else 'col'
            else:
                row_cnt, col_cnt = Counter(), Counter()
                for e in regex_edges[pii_t_idx]:
                    row_cnt[e[0][0]] += 1
                    col_cnt[e[0][1]] += 1
                orient[pii_t_idx] = 'row' if col_cnt.most_common(1)[0][1] >= row_cnt.most_common(1)[0][1] else 'col'

regex_edges = {pii_t_idx: regex_edges[pii_t_idx] for pii_t_idx in orient.keys()}

for pii_t_idx in regex_edges.keys():
    if orient[pii_t_idx] == 'row':
        regex_edges[pii_t_idx] = [e for e in regex_edges[pii_t_idx] if e[0][1] == e[1][1]]
    else:
        regex_edges[pii_t_idx] = [e for e in regex_edges[pii_t_idx] if e[0][0] == e[1][0]]

row_labels, col_labels = dict(), dict()
for pii_t_idx in regex_edges.keys():
    srcs = [e[0] for e in regex_edges[pii_t_idx]]
    rows, cols = set([x[0] for x in srcs]), set([x[1] for x in srcs])
    r, c = train_data_dict[pii_t_idx]['num_rows'], train_data_dict[pii_t_idx]['num_cols']
    row_labels[pii_t_idx], col_labels[pii_t_idx] = [0] * r, [0] * c
    if len(rows) == 0 and len(cols) == 0: continue
    if len(rows) == 1 and len(cols) == 1:
        src = regex_edges[pii_t_idx][0][0]
        if orient[pii_t_idx] == 'row':
            row_labels[pii_t_idx][src[0]] = 1
            col_labels[pii_t_idx][src[1]] = 2
        else:
            row_labels[pii_t_idx][src[0]] = 2
            col_labels[pii_t_idx][src[1]] = 1
    elif len(cols) == 1:
        for src, _ in regex_edges[pii_t_idx]:
            row_labels[pii_t_idx][src[0]] = 1
            col_labels[pii_t_idx][src[1]] = 2
    elif len(rows) == 1:
        for src, _ in regex_edges[pii_t_idx]:
            row_labels[pii_t_idx][src[0]] = 2
            col_labels[pii_t_idx][src[1]] = 1
    else:
        dst_cnt = Counter()
        for src, dst in regex_edges[pii_t_idx]:
            if src != dst:
                dst_cnt[dst] += 1
        regex_edges[pii_t_idx] = [e for e in regex_edges[pii_t_idx] if e[0] == e[1] or dst_cnt[e[1]] > 1]
        if len(regex_edges[pii_t_idx]) == 1: regex_edges[pii_t_idx] = []
        if orient[pii_t_idx] == 'row':
            for src, _ in regex_edges[pii_t_idx]:
                row_labels[pii_t_idx][src[0]] = 1
                col_labels[pii_t_idx][src[1]] = 2
        else:
            for src, _ in regex_edges[pii_t_idx]:
                row_labels[pii_t_idx][src[0]] = 2
                col_labels[pii_t_idx][src[1]] = 1

for pii_t_idx in regex_edges.keys():
    if len(regex_edges[pii_t_idx]) > 0:
        assert sum(row_labels[pii_t_idx] + col_labels[pii_t_idx]) > 0

train_data = pickle.load(open(os.path.join(table_dir, 'train_data_mcc_ci.pkl'), 'rb'))
train_data_dict = {(c['pii'], c['t_idx']): c for c in train_data}

for pii_t_idx in regex_edges.keys():
    if len(regex_edges[pii_t_idx]) == 0: continue
    c = train_data_dict[pii_t_idx]
    c['comp_table'] = True
    c['sum_less_100'] = 1
    c['row_label'], c['col_label'] = row_labels[pii_t_idx], col_labels[pii_t_idx]
    c['edge_list'] = []
    for src, dst in regex_edges[pii_t_idx]:
        c['edge_list'].append((src[0] * c['num_cols'] + src[1], dst[0] * c['num_cols'] + dst[1]))

pickle.dump(train_data, open(os.path.join(table_dir, 'train_data_mcc_pi.pkl'), 'wb'))
