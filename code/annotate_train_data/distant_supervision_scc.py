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

import numpy as np
from tqdm import tqdm


table_dir = '../../data'
train_data = pickle.load(open(os.path.join(table_dir, 'train_data_new.pkl'), 'rb'))
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


split_pii_t_idxs = pickle.load(open(os.path.join(table_dir, 'train_val_test_split.pkl'), 'rb'))
piis = list(set(pii for pii, _ in split_pii_t_idxs['train']))
print(len(piis))

num = r'(\d*\.\d+|\d+)'
comp_vars = ['x', 'y', 'z']
var = r'(' + r'|'.join(comp_vars) + r')'
enum = r'((' + num + var + r'+)|' + num + r'|' + var + r')'
expr = enum + r'(\s*[\+\-/]\s*' + enum + r')*'
expr = r'((' + expr + r')|(\(' + expr + r'\))|(' + num + r'\s*%))'


def x_pattern_1(constituents):
    sub_pat = expr + r'?\s*' + r'(' + r'|'.join(constituents) + r')'
    pat = re.compile(r'(' + sub_pat + r'(\s*[-*\:\,+]?\s*' + sub_pat + r')+)')
    return re.compile(sub_pat), pat


def x_parse_1(s, sub_pat, pat):
    for comp in re.findall(pat, s):
        nums_found = 0
        for l in re.findall(sub_pat, comp[0]):
            if l[0]: nums_found += 1
        if nums_found == 0: continue
        return True
    return False


def x_pattern_2(constituents):
    sub_pat = r'(' + r'|'.join(constituents) + r')\s*' + expr + r'?'
    pat = re.compile(r'(' + sub_pat + r'(\s*[-*\:\,+]?\s*' + sub_pat + r')+)')
    return re.compile(sub_pat), pat


def x_parse_2(s, sub_pat, pat):
    for comp in re.findall(pat, s):
        nums_found = 0
        for l in re.findall(sub_pat, comp[0]):
            if l[1]: nums_found += 1
        if nums_found == 0: continue
        return True
    return False


def x_pattern_3(constituents):
    sub_pat_1 = r'(' + r'|'.join(constituents) + r')\s*' + expr + r'?'
    sub_pat_2 = r'(' + sub_pat_1 + r'(\s*[-*\:\,+]?\s*' + sub_pat_1 + r')*)'
    sub_pat_2_ = sub_pat_1 + r'?(\s*[-*\:\,+]?\s*' + sub_pat_1 + r'?)*'
    sub_pat_3 = r'((' + sub_pat_2_ + r'|\(\s*' + sub_pat_2_ + r'\s*\)|\[\s*' + sub_pat_2_ + r'\s*\])\s*' + expr + r')'
    pat = re.compile(r'(' + sub_pat_3 + r'(\s*[-*\:\,+]?\s*' + sub_pat_3 + r')+)')
    sub_pat = [sub_pat_1, sub_pat_2, sub_pat_3]
    sub_pat = [re.compile(x) for x in sub_pat]
    return sub_pat, pat


def x_parse_3(s, sub_pat, pat):
    if ('(' not in s or ')' not in s) and ('[' not in s or ']' not in s):
        return False
    for comp in re.findall(pat, s):
        comp_s = re.sub(expr, ' ', comp[0])
        if ('(' not in comp_s or ')' not in comp_s) and ('[' not in comp_s or ']' not in comp_s): continue
        return True
    return False


def x_pattern_4(constituents):
    sub_pat_1 = expr + r'?\s*(' + r'|'.join(constituents) + r')'
    sub_pat_2 = r'(' + sub_pat_1 + r'(\s*[-*\:\,+]?\s*' + sub_pat_1 + r')*)'
    sub_pat_3 = r'((' + sub_pat_2 + r'|\(\s*' + sub_pat_2 + r'\s*\)|\[\s*' + sub_pat_2 + r'\s*\])\s*' + expr + r')'
    pat = re.compile(r'(' + sub_pat_3 + r'(\s*[-*\:\,+]?\s*' + sub_pat_3 + r')+)')
    sub_pat = [sub_pat_1, sub_pat_2, sub_pat_3]
    sub_pat = [re.compile(x) for x in sub_pat]
    return sub_pat, pat


def x_parse_4(s, sub_pat, pat):
    if ('(' not in s or ')' not in s) and ('[' not in s or ']' not in s):
        return False
    for comp in re.findall(pat, s):
        comp_s = re.sub(expr, ' ', comp[0])
        if ('(' not in comp_s or ')' not in comp_s) and ('[' not in comp_s or ']' not in comp_s): continue
        return True
    return False


def x_pattern_5(constituents):
    sub_pat_1 = r'(' + r'|'.join(constituents) + r')\s*' + expr + r'?'
    sub_pat_2 = r'(' + sub_pat_1 + r'(\s*[-*\:\,+]?\s*' + sub_pat_1 + r')*)'
    sub_pat_2_ = r'(' + sub_pat_1 + r'?(\s*[-*\:\,+]?\s*' + sub_pat_1 + r'?)*)'
    sub_pat_3 = r'(' + expr + '\s*(' + sub_pat_2_ + r'|\(\s*' + sub_pat_2_ + r'\s*\)|\[\s*' + sub_pat_2_ + '\s*\]))'
    pat = re.compile(r'(' + sub_pat_3 + r'(\s*[-*\:\,+]?\s*' + sub_pat_3 + r')+)')
    sub_pat = [sub_pat_1, sub_pat_2, sub_pat_3]
    sub_pat = [re.compile(x) for x in sub_pat]
    return sub_pat, pat


def x_parse_5(s, sub_pat, pat):
    if ('(' not in s or ')' not in s) and ('[' not in s or ']' not in s):
        return False
    for comp in re.findall(pat, s):
        comp_s = re.sub(expr, ' ', comp[0])
        if ('(' not in comp_s or ')' not in comp_s) and ('[' not in comp_s or ']' not in comp_s): continue
        return True
    return False


def x_pattern_6(constituents):
    sub_pat_1 = expr + r'?\s*(' + r'|'.join(constituents) + r')'
    sub_pat_2 = r'(' + sub_pat_1 + r'(\s*[-*\:\,+]?\s*' + sub_pat_1 + r')*)'
    sub_pat_3 = r'(' + expr + '\s*(' + sub_pat_2 + r'|\(\s*' + sub_pat_2 + r'\s*\)|\[\s*' + sub_pat_2 + '\s*\]))'
    pat = re.compile(r'(' + sub_pat_3 + r'(\s*[-*\:\,+]?\s*' + sub_pat_3 + r')+)')
    sub_pat = [sub_pat_1, sub_pat_2, sub_pat_3]
    sub_pat = [re.compile(x) for x in sub_pat]
    return sub_pat, pat


def x_parse_6(s, sub_pat, pat):
    if ('(' not in s or ')' not in s) and ('[' not in s or ']' not in s):
        return False
    for comp in re.findall(pat, s):
        comp_s = re.sub(expr, ' ', comp[0])
        if ('(' not in comp_s or ')' not in comp_s) and ('[' not in comp_s or ']' not in comp_s): continue
        return True
    return False


patterns = [x_pattern_1, x_pattern_2, x_pattern_3, x_pattern_4, x_pattern_5, x_pattern_6]
parses = [x_parse_1, x_parse_2, x_parse_3, x_parse_4, x_parse_5, x_parse_6]


def get_cons_pattern(gids, compositions):
    constituents = non_zero_cols(compositions.loc[gids])
    assert all(['-' not in c for c in constituents])
    assert len(gids) == 0 or len(constituents) > 0
    constituents = set(constituents) - set(['RO', 'R2O', 'R2O3'])
    constituents = sorted(constituents, key=lambda x: -len(x))
    constituents = [c.replace('(', '\(').replace(')', '\)') for c in constituents]
    return constituents, re.compile('|'.join(constituents), re.IGNORECASE)


def check_single_cell_comps(act_table, sub_pats, pats):
    res = []
    for r in act_table:
        res_r = []
        for s in r:
            res_c = False
            for sub_pat, pat, parse in zip(sub_pats, pats, parses):
                if parse(s.strip().replace('\n', ' '), sub_pat, pat):
                    res_c = True
                    break
            res_r.append(res_c)
        res.append(res_r)
    return res


def identify_good_tables(pii):
    xml_tables = pii_tables[pii]
    for table in xml_tables:
        table['regex_table'] = 0

    glass_ids = set(pii_glass_ids.loc[pii_glass_ids['PII'] == pii, 'GLASNO']) & avail_glass_ids
    if len(glass_ids) == 0: return
    constituents, _ = get_cons_pattern(glass_ids & gids['mol'], composition['mol'])
    if len(constituents) == 0: return

    sub_pats, pats = [], []
    for pattern, parse in zip(patterns, parses):
        sub_pat, pat = pattern(constituents)
        sub_pats.append(sub_pat)
        pats.append(pat)

    for table in xml_tables:
        scc_cell_labels = check_single_cell_comps(table['act_table'], sub_pats, pats)
        if np.array(scc_cell_labels).sum() > 0:
            table['regex_table'] = 1
            table['comp_table'] = True


for pii in tqdm(piis):
    identify_good_tables(pii)


pickle.dump(train_data, open(os.path.join(table_dir, 'train_data_scc.pkl'), 'wb'))
