#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created by luozhenyu on 2018/11/28
"""
from collections import deque
from simulator import data
import random


class RelayBuffer(object):
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.count = 0
        self.buffer = deque()
        for idx, row in data.iterrows():
            sample = list()
            sample.append(row['state_float'])
            sample.append(row['action_float'])
            sample.append(row['reward_float'])
            sample.append(row['n_state_float'])
            self.buffer.append(sample)

    def add(self, state, action, reward, next_reward):
        experience = (state, action, reward, next_reward)
        if self.count < self.buffer_size:
            self.buffer.append(experience)
            self.count += 1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)

    def size(self):
        return self.count

    def sample_batch(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def clear(self):
        self.buffer.clear()
        self.count = 0
