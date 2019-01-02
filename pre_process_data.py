#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created by luozhenyu on 2019/1/2
"""
import itertools
import pandas as pd
import numpy as np


SIGMA = 0.9


def calculate_reward(row):
    r = 0.
    for i, v in enumerate(row['reward'].split('|')):
        r += np.power(SIGMA, i) * (0 if v == "show" else 1)
    return r


def process_data(data_path, recall_path):
    data = pd.read_csv(data_path, sep='\t')
    for org in ["state", "action", "n_state"]:
        target = org + "_float"
        data[target] = data.apply(
            lambda row: [item for sublist in
                         list(map(lambda t: list(np.array(t.split(','), dtype=np.float64)), row[org].split('|')))
                         for item in sublist
                         ], axis=1
        )
    data['reward_float'] = data.apply(calculate_reward, axis=1)

    recall_data = pd.read_csv(recall_path, sep='\t')
    recall_data['embed_float'] = recall_data.apply(
        lambda row: np.array(row['embedding'][1:-1].split(','), dtype=np.float64).tolist(), axis=1
    )
    recall_tmp = list()
    for idx, row in recall_data.iterrows():
        for i in range(4):
            recall_tmp.append(row['embed_float'][i * 30: (i + 1) * 30])
    recall_tmp.sort()
    recall = list(l for l, _ in itertools.groupby(recall_tmp))

    return data, recall


data, recall_data = process_data("rl_tuple_data_path", "recall_data_path")
