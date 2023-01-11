import pandas as pd
import os

from utils import *

datasets = dict()
for split in splits:
    datasets[split] = TableDataset([comp_data_dict[pii_t_idx] for pii_t_idx in train_val_test_split[split]])

#directory where data is to be stored
par_dir = './data/ft_cell'
dir_name = 'discomat_data'
path = os.path.join(par_dir, dir_name)
os.makedirs(path, exist_ok=True)

#method to create csvs
def create_input_tables(split, no_of_tables):
    os.makedirs(os.path.join(path, f'discomat_{split}_csv'), exist_ok=True)
    for i in range(0,no_of_tables):
        tar_list = datasets[split][i]['act_table']
        pii = datasets[split][i]['pii']
        t_idx = datasets[split][i]['t_idx']
        new_name = pii + '__' + str(t_idx)
        df1 = pd.DataFrame(tar_list)
        file_name = f'{new_name}.csv'
        file_name = f'discomat_{split}_csv/{new_name}.csv'
        file_path = os.path.join(path, file_name)
        df1.to_csv(file_path, index=False, header=False)

#method to create labels
def create_labels(split, no_of_tables):
    dictt = {}
    tot_name, conc, tot_row_col_gid, tot_edge  = [], [], [], []
    for i in range(0, no_of_tables):
        tar_list = datasets[split][i]['act_table']
        pii = datasets[split][i]['pii']
        t_idx = datasets[split][i]['t_idx']
        new_name = pii + '__' + str(t_idx)
        row_l = datasets[split][i]['row_label']
        col_l = datasets[split][i]['col_label']
        conc = row_l + col_l
        tot_row_col_gid += [conc]
        tot_name += [new_name]
        edg_l = datasets[split][i]['edge_list']
        tot_edge += [edg_l]

    dictt['fname'] = tot_name
    dictt['row_id'] = tot_row_col_gid
    dictt['col_id'] = tot_edge
    file_name = f'discomat_{split}_label.csv'
    file_path = os.path.join(path, file_name)
    df2 = pd.DataFrame(dictt).to_csv(file_path, index = False)
    
#method to create table_list
os.makedirs('./data/table_list', exist_ok=True) 
def create_list(split, no_of_tables):
    table_list = []
    for i in range(0, no_of_tables):
        pii = datasets[split][i]['pii']
        t_idx = datasets[split][i]['t_idx']
        new_name = pii + '__' + str(t_idx)
        table_list.append(new_name)
    pickle.dump(table_list, open(f'./data/table_list/{split}_list.pkl', 'wb'))

#generating csvs
create_input_tables('train', 3387)
create_input_tables('val', 519)
create_input_tables('test', 512)

#generating labels
create_labels('train', 3387)
create_labels('val', 519)
create_labels('test', 512)

#generating table_list
create_list('train', 3387)
create_list('val', 519)
create_list('test', 512)