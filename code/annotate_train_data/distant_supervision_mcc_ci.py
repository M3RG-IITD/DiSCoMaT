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

from collections import defaultdict
import os
import pickle
import re

from tqdm import tqdm


table_dir = '../../data'

train_data = pickle.load(open(os.path.join(table_dir, 'train_data_scc.pkl'), 'rb'))
pii_glass_ids = pickle.load(open(os.path.join(table_dir, 'sciglass_pii_gids.pkl'), 'rb'))

train_data = sorted(train_data, key=lambda x: (x['pii'], x['t_idx']))
pii_tables = defaultdict(list)

for c in train_data:
    pii_tables[c['pii']].append(c)


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

print(len(avail_glass_ids))


def viewcomp(df):
    return df.T[df.astype(bool).sum(axis=0) > 0].T


split_pii_t_idxs = pickle.load(open(os.path.join(table_dir, 'train_val_test_split.pkl'), 'rb'))
piis = list(set(pii for pii, _ in split_pii_t_idxs['train']))
print(len(piis))


num_pattern = re.compile(r'\d*\.\d+|\d+')
comp_lower_dict = {c.lower(): c for c in composition['wt'].columns}

def comp_post_process(comp):
    return comp_lower_dict[comp.lower()]

def num_post_process(num):
    return float(num)

def get_comp_and_nums(table, cons_pattern):
    comps, nums = [], []
    for r in table:
        r_comps, r_nums = [], []
        for cell in r:
            cell_comps = re.findall(cons_pattern, cell)
            r_comps.append(list(set(map(comp_post_process, cell_comps))))
            cell_nums = re.findall(num_pattern, re.sub(cons_pattern, ' ', cell))
            r_nums.append(list(set(map(num_post_process, cell_nums))))
        comps.append(r_comps)
        nums.append(r_nums)
    return comps, nums


def l1_dist(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def closest(l, p):
    res = l[0]
    for x in l:
        if l1_dist(x, p) < l1_dist(res, p):
            res = x
    return res

def within_tol(num, l, tol):
    if not l: return False
    lo, hi = (1 - tol) * num, (1 + tol) * num
    return any(lo <= n <= hi for n in l)

def check_row_and_col(n, nums, row_num, col_num, tol):
    res = []
    tmp = [row_num, col_num]
    if within_tol(n, nums[row_num][col_num], tol):
        res.append(tmp + [row_num, col_num])
    for j in range(len(nums[row_num])):
        if j == col_num: continue
        if within_tol(n, nums[row_num][j], tol):
            res.append(tmp + [row_num, j])
    for i in range(len(nums)):
        if i == row_num: continue
        if within_tol(n, nums[i][col_num], tol):
            res.append(tmp + [i, col_num])
    return res

def get_max_comp_row_col(t):
    ans = 0
    row_len, col_len = [], []
    for i in range(len(t)):
        s = set()
        for j in t[i]:
            s |= set(j.keys())
        row_len.append(len(s))
    
    for j in range(len(t[0])):
        s = set()
        for i in range(len(t)):
            s |= set(t[i][j].keys())
        col_len.append(len(s))
    
    max_len = max(row_len + col_len)
    idxs = {
        'row': [i for i, x in enumerate(row_len) if x == max_len],
        'col': [i for i, x in enumerate(col_len) if x == max_len],
    }
    return max_len, idxs

def get_comp_location_in_table(t, orient, idx):
    assert orient in ['row', 'col']
    if orient == 'row':
        return {j: closest(list(t[idx][j].values()), (idx, j)) for j in range(len(t[idx])) if len(t[idx][j]) > 0}
    else:
        return {i: closest(list(t[i][idx].values()), (i, idx)) for i in range(len(t)) if len(t[i][idx]) > 0}

def search_db_row_in_table(db_row, comps, nums):
    comp_locations = defaultdict(list)
    for i, r in enumerate(comps):
        for j, c in enumerate(r):
            for comp in c:
                comp_locations[comp].append((i, j))

    res = {'row': {}, 'col': {}}
    for tol in [1e-4, 1e-2, 2e-2, 3e-2]:
        num_comp_total = 0
        num_loc = []
        for i in range(len(comps)):
            num_loc.append([dict() for j in range(len(comps[0]))])
        for c, n in db_row.iteritems():
            if n == 0: continue
            num_comp_total += 1
            comp_res = []
            for cx, cy in comp_locations[c]:
                comp_res += check_row_and_col(n, nums, cx, cy, tol)
            for cx, cy, i, j in comp_res:
                if c in num_loc[i][j]:
                    if l1_dist((i, j), (cx, cy)) < l1_dist((i, j), num_loc[i][j][c]):
                        num_loc[i][j][c] = (cx, cy)
                else:
                    num_loc[i][j][c] = (cx, cy)

        max_len, idxs = get_max_comp_row_col(num_loc)
        if max_len >= max(num_comp_total - 1, 2):
            for k in idxs.keys(): # ['row', 'col']
                for idx in idxs[k]:
                    res[k][idx] = get_comp_location_in_table(num_loc, k, idx)
            return res
    return res

def search_glass_id_in_table(db_row, comps, nums, table):
    d1, d2 = search_db_row_in_table(db_row, comps, nums), search_db_row_in_table(db_row * 0.01, comps, nums)
    res = dict()
    for k in d1.keys():  # ['row', 'col']
        if len(d1[k]) == 1 and len(d2[k]) == 1:
            k1, k2 = list(d1[k].keys())[0], list(d2[k].keys())[0]
            assert type(d1[k][k1]) == dict and type(d2[k][k2]) == dict
            if len(d1[k][k1]) >= len(d2[k][k2]):
                res[k] = d1[k]
            else:
                res[k] = d2[k]
        elif len(d1[k]) >= len(d2[k]):
            res[k] = d1[k]
        else:
            res[k] = d2[k]
    return res

def search_glass_id_in_paper(db_row, paper_tables, comp_pattern):
    res = []
    for table in paper_tables:
        if table['regex_table'] == 1:
            res.append({'row': {}, 'col': {}})
            continue
        comps, nums = get_comp_and_nums(table['act_table'], comp_pattern)
        res.append(search_glass_id_in_table(db_row, comps, nums, table['act_table']))
    return res

def get_cons_pattern(gids, compositions):
    constituents = non_zero_cols(compositions.loc[gids])
    assert all(['-' not in c for c in constituents])
    constituents = set(constituents) - set(['RO', 'R2O', 'R2O3'])
    constituents = sorted(constituents, key=lambda x: -len(x))
    constituents = [c.replace('(', '\(').replace(')', '\)') for c in constituents]
    return constituents, re.compile('|'.join(constituents), re.IGNORECASE)

def uniq_list(l):
    res = []
    l = sorted(l, key=lambda x: -len(x))
    for x in l:
        flag = True
        for r in res:
            if x.items() <= r.items():
                flag = False
                break
        if flag: res.append(x)
    return res

def get_cols(d):
    cols = set()
    for i in d.keys():
        assert type(d[i]) == list
        for x in d[i]:
            assert type(x) == dict
            cols |= set(x.keys())
    return cols
    
def search_glass_ids_in_paper(pii):
    xml_tables = pii_tables[pii]
    for t in xml_tables:
        for k in composition.keys(): # mol, wt
            t[k] = {'row': defaultdict(list), 'col': defaultdict(list)}

    glass_ids = set(pii_glass_ids.loc[pii_glass_ids['PII'] == pii, 'GLASNO']) & avail_glass_ids
    if len(glass_ids) == 0: return
    wt_mol_info = defaultdict(dict)
    for k in composition.keys():
        wt_mol_info[k]['gids'] = glass_ids & gids[k]
        wt_mol_info[k]['compounds'], wt_mol_info[k]['cons_pattern'] = get_cons_pattern(wt_mol_info[k]['gids'], composition[k])
        wt_mol_info[k]['db'] = viewcomp(composition[k].loc[wt_mol_info[k]['gids']])

    for gid in glass_ids:
        for k in composition.keys():  # ['mol', 'wt']
            if gid in wt_mol_info[k]['gids']:
                res = search_glass_id_in_paper(wt_mol_info[k]['db'].loc[gid], xml_tables, wt_mol_info[k]['cons_pattern'])
                for t, r in zip(xml_tables, res):
                    for k_ in r.keys():  # ['row', 'col']
                        for x in r[k_]:
                            assert type(x) == int
                            t[k][k_][x].append(r[k_][x])

    for t in xml_tables:
        for k in composition.keys():
            if len(t[k]['row']) == 0 and len(t[k]['col']) == 0:
                t[k]['row'] = None
                t[k]['col'] = None
            elif len(t[k]['row']) >= len(t[k]['col']):
                t[k]['col'] = None
                for x in t[k]['row']:
                    t[k]['row'][x] = uniq_list(t[k]['row'][x])
            else:
                t[k]['row'] = None
                for x in t[k]['col']:
                    t[k]['col'][x] = uniq_list(t[k]['col'][x])

        if t['mol']['row'] and t['wt']['col']:
            if len(t['mol']['row']) >= len(t['wt']['col']):
                t['wt']['col'] = None
            else:
                t['mol']['row'] = None
        
        elif t['mol']['col'] and t['wt']['row']:
            if len(t['mol']['col']) >= len(t['wt']['row']):
                t['wt']['row'] = None
            else:
                t['mol']['col'] = None
        
        elif t['mol']['row'] and t['wt']['row']:
            if len(set(t['mol']['row'].keys()) & set(t['wt']['row'].keys())) > 0 and \
            len(get_cols(t['mol']['row']) & get_cols(t['wt']['row'])) > 0:
                if len(t['mol']['row']) >= len(t['wt']['row']):
                    t['wt']['row'] = None
                else:
                    t['mol']['row'] = None
        
        elif t['mol']['col'] and t['wt']['col']:
            if len(set(t['mol']['col'].keys()) & set(t['wt']['col'].keys())) > 0 and \
            len(get_cols(t['mol']['col']) & get_cols(t['wt']['col'])) > 0:
                if len(t['mol']['col']) >= len(t['wt']['col']):
                    t['wt']['col'] = None
                else:
                    t['mol']['col'] = None


for pii in tqdm(piis):
    search_glass_ids_in_paper(pii)


def populate_comp_labels(t: dict):
    # 0 -> no label
    # 1 -> composition present
    # 2 -> constituent prsesent
    t['row_label'], t['col_label'] = [0] * t['num_rows'], [0] * t['num_cols']
    for k in ['mol', 'wt']:
        if t[k]['row']:
            for x in t[k]['row'].keys():
                t['row_label'][x] = 1
            for x in get_cols(t[k]['row']):
                t['col_label'][x] = 2
        elif t[k]['col']:
            for x in t[k]['col'].keys():
                t['col_label'][x] = 1
            for x in get_cols(t[k]['col']):
                t['row_label'][x] = 2
    t['comp_table'] = sum(t['row_label'] + t['col_label']) > 0


def populate_edges(t: dict):
    t['edge_list'] = []
    
    def get_node_num(n):
        return n[0] * t['num_cols'] + n[1]
    
    for k in ['mol', 'wt']:
        for orient in ['row', 'col']:
            if t[k][orient] is None: continue
            for idx in t[k][orient]:
                for x in t[k][orient][idx]:
                    for a in x:
                        src = (idx, a) if orient == 'row' else (a, idx)
                        dst = x[a]
                        t['edge_list'].append((get_node_num(src), get_node_num(dst)))
    t['edge_list'] = list(set(t['edge_list']))


def populate_mol_wt_labels(t: dict):
    # 0 -> no label
    # 1 -> mol
    # 2 -> wt
    t['mol_wt'] = [[0] * t['num_cols'] for _ in range(t['num_rows'])]
    for label, k in zip([1, 2], ['mol', 'wt']):
        for orient in ['row', 'col']:
            if t[k][orient] is None: continue
            for idx in t[k][orient]:
                for x in t[k][orient][idx]:
                    for a in x:
                        src = (idx, a) if orient == 'row' else (a, idx)
                        t['mol_wt'][src[0]][src[1]] = label

for pii in piis:
    for t in pii_tables[pii]:
        populate_comp_labels(t)
        populate_edges(t)
        # populate_mol_wt_labels(t)
        t.pop('mol')
        t.pop('wt')

pickle.dump(train_data, open(os.path.join(table_dir, 'train_data_mcc_ci.pkl'), 'wb'))
