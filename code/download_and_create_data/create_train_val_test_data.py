import os
import pickle
import sys
sys.path.append('..')

from transformers import AutoTokenizer
from tqdm import tqdm

from normalize_text import normalize


table_dir = '../../data'
split_pii_t_idxs = pickle.load(open(os.path.join(table_dir, 'train_val_test_split.pkl'), 'rb'))
pii_table_dict = pickle.load(open(os.path.join(table_dir, 'pii_table_dict.pkl'), 'rb'))

lm_name = 'm3rg-iitd/matscibert'
cache_dir = os.path.join(table_dir, '.cache')
tokenizer = AutoTokenizer.from_pretrained(lm_name, cache_dir=cache_dir, model_max_length=512)


for pii in tqdm(pii_table_dict):
    for t_idx, d in enumerate(pii_table_dict[pii]):
        d['pii'] = pii
        d['t_idx'] = t_idx
        d['num_rows'] = len(d['act_table'])
        d['num_cols'] = len(d['act_table'][0])
        d['num_cells'] = d['num_rows'] * d['num_cols']
        d['comp_table'] = False
        d['sum_less_100'] = 0
        
        table_vec = []
        for row in d['act_table']:
            table_vec += row
        table_vec = [normalize(cell) for cell in table_vec]
        tok = tokenizer(table_vec, max_length=50, truncation=True)
        d['input_ids'] = tok['input_ids']
        d['attention_mask'] = tok['attention_mask']

        tok = tokenizer([normalize(d['caption'])])
        d['caption_input_ids'] = tok['input_ids']
        d['caption_attention_mask'] = tok['attention_mask']
        assert len(d['caption_input_ids'][0]) <= 512


splits = ['train', 'val', 'test']
data = {split: [] for split in splits}

for split in splits:
    for pii, t_idx in split_pii_t_idxs[split]:
        data[split].append(pii_table_dict[pii][t_idx])

pickle.dump(data['train'], open(os.path.join(table_dir, 'train_data_new.pkl'), 'wb'))
pickle.dump(data['val'] + data['test'], open(os.path.join(table_dir, 'val_test_data_new.pkl'), 'wb'))
