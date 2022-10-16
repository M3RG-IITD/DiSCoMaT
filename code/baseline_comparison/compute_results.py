from argparse import ArgumentParser
import os
import pickle

import numpy as np
import pandas as pd


parser = ArgumentParser()
parser.add_argument('--model', choices=['tapas', 'tapas_adapted', 'tabert', 'tabert_adapted', 'gat'], required=True, type=str)
parser.add_argument('--split', choices=['val', 'test'], default='test', type=str)
args = parser.parse_args()

table_dir = '../../data'

violation_keys = ['1_2_violations', '1_3_violations', '2_3_violations', '3_3_violations']


def get_results(model):
    mid_fscores, tuple_metrics, mat_metrics, violations = [], [], [], []
    for seed in range(3):
        res_file = os.path.join(table_dir, 'res_dir', f'res_baseline_{model}_{seed}.pkl')
        res = pickle.load(open(res_file, 'rb'))[args.split]
        mid_fscores.append(res['comp_gid_stats'].iloc[-1, -1])
        tuple_metrics.append(res['tuple_metrics'])
        mat_metrics.append(res['composition_metrics'])
        violations.append(sum(res[k] for k in violation_keys))
    
    print('MID F1-score')
    print(round(np.mean(mid_fscores) * 100, 2), round(np.std(mid_fscores, ddof=1) * 100, 2))
    print()
    
    tuple_df = pd.DataFrame(tuple_metrics)
    mat_df = pd.DataFrame(mat_metrics)
    mean_df = pd.concat([tuple_df.mean(), mat_df.mean()], axis=1).T
    mean_df.index = ['Tuple', 'Mat']
    print('mean')
    print(mean_df)

    std_df = pd.concat([tuple_df.std(ddof=1), mat_df.std(ddof=1)], axis=1).T
    std_df.index = ['Tuple', 'Mat']
    print('std')
    print(std_df)
    print()
    
    print('CV')
    print(round(np.mean(violations), 2), round(np.std(violations, ddof=1), 2))
    print()


get_results(args.model)
