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


def gen_samples(id_num=100, sample_size=256):
    from pandas import DataFrame
    ids = np.random.randint(0, 100, size=id_num)
    ids = [str(idx) for idx in ids]
    embeddings = np.random.randn(id_num, 30)
    id_emb_dic = dict(zip(ids, embeddings))
    colunms_name = ['state', 'action', 'reward', 'n_state', 'recall']
    sample_data = []
    for i in range(sample_size):
        state_len = np.random.randint(1, 12)
        state = [str(val) for val in np.random.choice(ids, size=state_len)]
        n_state = [str(val) for val in np.random.choice(ids, size=state_len)]
        action = str(np.random.choice(ids, size=2)[0])
        reward = np.random.rand()
        recall = [action]
        sample_data.append((state, action, reward, n_state, recall))
    data = DataFrame(sample_data, columns=colunms_name)
    write_file(id_emb_dic, sample_data)
    return id_emb_dic, data


def write_file(embedding_file, sample_data):
    with open("embed.csv", "w") as fout:
        head = 'item_id\tembedding\n'
        fout.write(head)
        for item_id, emb in embedding_file.items():
            emb_str = ','.join([str(v) for v in emb])
            outline = '%s\t%s\n' % (item_id, emb_str)
            fout.write(outline)
    print("wrote embedding done")

    with open("train.csv", "w") as fout_sample:
        columns_name = ['state', 'action', 'reward', 'n_state', 'recall']
        head = '%s\n' % ('\t'.join(columns_name))
        fout_sample.write(head)
        for sample in sample_data:
            s_state = '|'.join(sample[0])
            action = sample[1]
            s_reward = str(sample[2])
            s_n_state = '|'.join(sample[3])
            s_recall = '|'.join(sample[4])
            outline = '{}\t{}\t{}\t{}\t{}\n'.format(s_state, action, s_reward, s_n_state, s_recall)
            fout_sample.write(outline)
    print("wrote sample data done")


data, recall_data = process_data("train.csv", "embed.csv")
