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

