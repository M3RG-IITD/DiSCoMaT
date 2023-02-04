from collections import defaultdict
import os
import pickle
import re

import numpy as np

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from tqdm import tqdm


table_dir = '../../data'
comp_data = pickle.load(open(os.path.join(table_dir, 'val_test_data.pkl'), 'rb'))

train_val_test_split = pickle.load(open(os.path.join(table_dir, 'train_val_test_split.pkl'), 'rb'))
comp_data_dict = {(c['pii'], c['t_idx']): c for c in comp_data}

splits = ['train', 'val', 'test']

for split in splits[1:3]:
    train_val_test_split[split] = [x for x in train_val_test_split[split] if comp_data_dict[x]['regex_table'] == 0 and comp_data_dict[x]['sum_less_100'] == 0]
    # regex table == 0 are non sccs and sum_less_100 == 0 are mcc_ci

for split in splits[1:3]:
    for pii_t_idx in train_val_test_split[split]:
        c = comp_data_dict[pii_t_idx]
        if sum(c['gid_row_label']) == 1:
            c['row_label'][c['gid_row_label'].index(1)] = 3
        elif sum(c['gid_col_label']) == 1:
            c['col_label'][c['gid_col_label'].index(1)] = 3

pii_tables = defaultdict(list)
for split in splits[1:3]:
    for pii_tid in train_val_test_split[split]:
        pii_tables[pii_tid[0]].append(comp_data_dict[pii_tid])

all_elements = pickle.load(open(os.path.join(table_dir, 'elements_compounds.pkl'), 'rb'))['elements']
all_compounds = pickle.load(open(os.path.join(table_dir, 'elements_compounds.pkl'), 'rb'))['compounds']
all_compounds_new = list(map(lambda s: s.replace('\\', ''), all_compounds))


comp_num_pattern = r'(\d+\.\d+|\d+/\d+|\d+)'
ele_num = r'((' + '|'.join(all_elements) + r')' + comp_num_pattern + r'?)'
many_ele_num = r'(' + ele_num + r')+'
ele_comp_pattern = r'(((\(' + many_ele_num + '\)' + comp_num_pattern + r')|(' + many_ele_num + r'))+)'
ele_comp_pattern = r'(?:^|\W)(' + ele_comp_pattern + r'|Others?)(?:\W|$)'

ele_pattern = r'(' + '|'.join(all_elements) + r')'
cpd_pattern = r'(' + '|'.join(all_compounds) + r')'
num_pattern = r'(\d+\.\d+|\d+)'

re.findall(ele_comp_pattern, "Ca(NO3)2")
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

def num_post_process(num):
    return float(num)

def find_num(string):
    #e.g. already in int or float form: 12.5 -> 12.5
    try:
        return float(string)
    except:
        pass
    #e.g. 12.5 - 13.5 -> 13.0
    range_regex = re.compile('\d+\.?\d*\s*-\s*\d+\.?\d*')
    try:
        ranges = range_regex.search(string).group().split('-')
        num = float(ranges[0])
        return num
    except:
        pass
    #e.g. 12.2 (5.2) -> 12.2
    bracket_regex = re.compile('(\d+\.?\d*)\s*\(\d*.?\d*\)')
    try:
        extracted_value = float(bracket_regex.search(string).group(1))
        return float(extracted_value)
    except:
        pass
    #e.g. 12.3 ± 0.5 -> 12.3
    plusmin_regex = re.compile('(\d+\.?\d*)(\s*[±+-]+\s*\d+\.?\d*)')  
    try:
        extracted_value = float(plusmin_regex.search(string).group(1))
        return extracted_value
    except AttributeError:
        pass
    #e.g. <0.05 -> 0.05  |  >72.0 -> 72.0    | ~12 -> 12
    lessthan_roughly_regex = re.compile('([<]|[~]|[>])=?\s*\d+\.*\d*')
    try:
        extracted_value = lessthan_roughly_regex.search(string).group()
        num_regex = re.compile('\d+\.*\d*')
        extracted_value = num_regex.search(extracted_value).group()
        return float(extracted_value)
    except:
        pass
    # e.g. 0.4:0.6 (ratios)
    if ':' in string:
        split = string.split(":")
        try:
            extracted_value = round(float(split[0])/float(split[1]), 3)
            return extracted_value
        except:
            pass
    return None

def get_nums(table):
    nums = []
    for r in table:
        r_nums = []
        for cell in r:
            # cell_nums = re.findall(num_pattern, re.sub(cons_pattern, ' ', cell))
            # cell_nums = re.findall(num_pattern, cell)
            num = find_num(cell)
            if num != None:
                cell_nums = [find_num(cell)]
            else:
                cell_nums = []
            r_nums.append(list(set(map(num_post_process, cell_nums))))
        nums.append(r_nums)
    return nums

remove_list = all_elements
def get_compounds_elements(table):
    constituents = []
    for r in table:
        r_constituents = []
        for cell in r:
            cell_constituents = re.findall(ele_comp_pattern, cell)
            cell_constituents_new = []
            for i, tup in enumerate(cell_constituents):
                cell_constituents_new += [e for e in tup if e!='']
            cell_constituents_new = list(set(cell_constituents_new)) # remove duplicates
            if len(cell_constituents_new) != 0:
                cell_constituents_new = [max(cell_constituents_new, key=len)]
                if cell_constituents_new[0] in remove_list:
                    cell_constituents_new = []
            r_constituents.append(cell_constituents_new) 
        constituents.append(r_constituents)
    return constituents

def get_orientation(t):
    table = pd.DataFrame(t['act_table'])
    r1 = table.iloc[0].values    
    r2 = table.iloc[1].values
    c1 = table[0].values
    c2 = table[1].values
    loopables = [r1,r2,c1,c2]
    counts = len(loopables)*[0]
    for i,loopable in enumerate(loopables):
        for text in loopable:
            for compound in all_compounds_new:
                if compound in text.strip():
                    counts[i] = counts[i]+1
    if np.argmax(counts)<2:
        orientation= 'c'    
    else:
        orientation= 'r'    
    return orientation

def get_gold_tuples(pii, t_idx):
    '''
    Function for obtaining gold tuples using paper PII and table index
    '''
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

mid_terms_list = ['sample', 'code', 'system', 'glass', 'id', 'no.', 'group', 'name', 'x', 'y', 'label']

def get_pred_tuples(table):
    tuples = []
    r, c = table['num_rows'], table['num_cols']
    pii, t_idx = table['pii'], table['t_idx']
    # orientation = get_orientation(table)
    # identify compounds in row headers - will be either in row 0 or 1
    row_0_ratio, row_1_ratio = 0, 0
    for cell in table['table_compounds_elements'][0]:
        if len(cell) != 0:
            row_0_ratio += 1
    for cell in table['table_compounds_elements'][1]:
        if len(cell) != 0:
            row_1_ratio += 1
    header_row = 0 if row_0_ratio > row_1_ratio else 1
    column_indexes = []
    constituents = []
    for i, cell in enumerate(table['table_compounds_elements'][header_row]):
        if len(cell) != 0 and cell[0] in all_compounds_new:
            column_indexes.append(i)
            constituents.append(cell[0])
    # identify material ids
    mids_present = False
    for term in mid_terms_list:
        if term in table['act_table'][0][0].lower() or term in table['act_table'][1][0].lower():
            mids_present = True
    row_indexes = []
    mids = []
    for i in range(header_row+1, r):
        row_indexes.append(i)
        if mids_present:
            mids.append(table['act_table'][i][0])
        else:
            mids.append('')
    mid_index = 0 if not mids_present else 1
    for i_ind, i in enumerate(row_indexes):
        k = 0
        for j_ind, j in enumerate(column_indexes):
            if len(table['table_nums'][i][j]) == 1 or table['act_table'][i][j] == '_':
                if len(table['table_nums'][i][j]) == 1:
                    x = table['table_nums'][i][j][0]
                else:
                    x = 0
                # prefix = f'{pii}_{t_idx}_{i}_{j}_{k}'
                prefix = f'{pii}_{t_idx}_{i}_{mid_index}_{k}'
                if mids[i_ind] != '':
                    gid = prefix + '_' + mids[i_ind]
                else:
                    gid = prefix
                unit = pred_cell_mol_wt(table, i, j)
                if x != 0:
                    tuples.append((gid, constituents[j_ind], round(float(x), 5), unit))
        k += 1
    return tuples

def get_tuples_metrics(gold_tuples_new, pred_tuples_new):
    gold_tuples = gold_tuples_new
    pred_tuples = pred_tuples_new
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
    extra_preds = []
    for p, v in pred_comps.items():
        found = False
        if p in gold_comps and gold_comps[p] == v:
            prec += 1
            found = True
        if not found:
            fi = p.find('_')
            pii = p[:fi]
            tid = int(p[fi+1:p.find('_', fi+1)])
            extra_preds.append((pii, tid))
    if len(pred_comps) > 0:
        prec /= len(pred_comps)
    else:
        prec = 0.0
    rec = 0
    missed_preds = []
    for g, v in gold_comps.items():
        found = False
        if g in pred_comps and pred_comps[g] == v:
            rec += 1
            found = True
        if not found:
            fi = g.find('_')
            pii = g[:fi]
            tid = int(g[fi+1:g.find('_', fi+1)])
            missed_preds.append((pii, tid))
    rec /= len(gold_comps)
    fscore = 2 * prec * rec / (prec + rec) if (prec + rec > 0) else 0.0
    metrics = {'precision': prec, 'recall': rec, 'fscore': fscore}
    metrics = {m: round(v * 100, 2) for m, v in metrics.items()}
    return metrics, extra_preds, missed_preds

def is_comp_table(table):
    if 'metallic' in str(table['caption']).lower():
        return 0
    elif 'decomposition' in str(table['caption']).lower():
        return 0
    elif ' composition' in str(table['caption']).lower():
        return 1
    elif 'composition' in str(table['caption']).lower():
        return 1
    elif 'composition' in str(table['act_table']).lower():
        return 1
    elif 'recipe' in str(table['act_table']).lower():
        return 1
    elif '-x' in str(table['caption']).replace('\n',''):
        return 1
    else:
        return 0

        
all_gold_tuples, all_pred_tuples = [], []
all_gold_mids, all_pred_mids = [], []

for i, pii in enumerate(list(pii_tables.keys())):

    for table in pii_tables[pii]:
        r, c = table['num_rows'], table['num_cols']
        gold_tuples = get_gold_tuples(pii, table['t_idx'])
        all_gold_tuples += gold_tuples
        all_gold_mids += table['gid_row_label'] + table['gid_col_label']
        pred_mids = [0 for _ in range(r+c)]
        pred_tuples = []
        
        if is_comp_table(table):
            # assume simple col orientation (orientation detection performs poorer)
            table['table_nums'] = get_nums(table['act_table'])
            table['table_compounds_elements'] = get_compounds_elements(table['act_table'])

            pred_tuples = get_pred_tuples(table)
            all_pred_tuples += pred_tuples
            if len(pred_tuples) != 0:
                pred_mids[r] = 1 # always predicts first column as mid
                
        all_pred_mids += pred_mids
            
mid_fscore = f1_score(all_gold_mids, all_pred_mids)
mid_precision = precision_score(all_gold_mids, all_pred_mids)
mid_recall = recall_score(all_gold_mids, all_pred_mids)
mid_accuracy = accuracy_score(all_gold_mids, all_pred_mids)

tuple_metrics = get_tuples_metrics(all_gold_tuples, all_pred_tuples)
mat_metrics, extra_preds, missed_preds = get_composition_metrics(all_gold_tuples, all_pred_tuples)

print(f'mid_metrics fscore {round(mid_fscore*100, 2)} precision {round(mid_precision*100, 2)} recall {round(mid_recall*100, 2)} accuracy {round(mid_accuracy*100, 2)}')
print(f'tuple_metrics {tuple_metrics}')
print(f'mat_metrics {mat_metrics}')