from collections import defaultdict
import pickle
import os
import sys
sys.path.append('..')

from tqdm import tqdm

from regex_lib import parse_composition


table_dir = '../../data'
val_test_data = pickle.load(open(os.path.join(table_dir, 'val_test_data_new.pkl'), 'rb'))
data_dict = {(c['pii'], c['t_idx']): c for c in val_test_data}
text_data = pickle.load(open(os.path.join(table_dir, 'train_val_test_paper_data.pkl'), 'rb'))


extracted_regex = defaultdict(dict)
for pii, t_idx in tqdm(data_dict.keys()):
    if pii in text_data:
        for section, text in text_data[pii].items():
            extracted_regex[pii][section] = parse_composition(text)
    else:
        extracted_regex[pii]['Title'] = []
        extracted_regex[pii]['Abstract'] = []
    c = data_dict[(pii, t_idx)]
    extracted_regex[pii][t_idx] = parse_composition(c['caption'].replace('\n', ' '))
    extracted_regex[pii][f'{t_idx}_footer'] = []
    for f in c['footer'].values():
        extracted_regex[pii][f'{t_idx}_footer'] += parse_composition(f)


pickle.dump(extracted_regex, open(os.path.join(table_dir, 'extracted_regex.pkl'), 'wb'))
