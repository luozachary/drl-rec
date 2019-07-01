#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created by luozhenyu on 2018/11/26
"""
from pre_process_data import data
import numpy as np


class Simulator(object):
    def __init__(self, alpha=0.5, sigma=0.9):
        self.data = data
        self.alpha = alpha
        self.sigma = sigma
        self.init_state = self.reset()
        self.current_state = self.init_state
        self.rewards, self.group_sizes, self.avg_states, self.avg_actions = self.avg_group()

    def reset(self):
        init_state = np.array(self.data['state_float'].sample(1).values[0]).reshape((1, 12, 30))
        self.current_state = init_state
        return init_state

    def step(self, action):
        simulate_rewards, result = self.simulate_reward((self.current_state.reshape((1, 360)),
                                                         action.reshape((1, 120))))
        for i, r in enumerate(simulate_rewards.split('|')):
            if r != "show":
                # self.current_state.append(action[i])
                tmp = np.append(self.current_state[0], action[i].reshape((1, 30)), axis=0)
                tmp = np.delete(tmp, 0, axis=0)
                self.current_state = tmp[np.newaxis, :]
        return result, self.current_state

    def avg_group(self):
        """calculate average state/action value for each group."""
        rewards = list()
        avg_states = list()
        avg_actions = list()
        group_sizes = list()
        for reward, group in self.data.groupby(['reward']):
            n_size = group.shape[0]
            state_values = group['state_float'].values.tolist()
            action_values = group['action_float'].values.tolist()
            avg_states.append(
                np.sum(state_values / np.linalg.norm(state_values, 2, axis=1)[:, np.newaxis], axis=0) / n_size
            )
            avg_actions.append(
                np.sum(action_values / np.linalg.norm(action_values, 2, axis=1)[:, np.newaxis], axis=0) / n_size
            )
            group_sizes.append(n_size)
            rewards.append(reward)
        return rewards, group_sizes, avg_states, avg_actions

    def simulate_reward(self, pair):
        """use the average result to calculate simulated reward.
        Args:
            pair (tuple): <state, action> pair

        Returns:
            simulated reward for the pair.
        """
        probability = list()
        denominator = 0.
        max_prob = 0.
        result = 0.
        simulate_rewards = ""
        # change a different way to calculate simulated reward
        for s, a, r in zip(self.avg_states, self.avg_actions, self.rewards):
            numerator = self.alpha * (
                np.dot(pair[0], s)[0] / (np.linalg.norm(pair[0], 2) * np.linalg.norm(s, 2))
            ) + (1 - self.alpha) * (
                np.dot(pair[1], a)[0] / (np.linalg.norm((pair[1], 2) * np.linalg.norm(a, 2)))
            )
            probability.append(numerator)
            denominator += numerator
            if numerator > max_prob:
                max_prob = numerator
                simulate_rewards = r
        probability /= denominator
        for p, r in zip(probability, self.rewards):
            for k, reward in enumerate(r.split('|')):
                result += p * np.power(self.sigma, k) * (0 if reward == "show" else 1)

        # calculate simulated reward by group
        # for i, reward in enumerate(self.rewards):
        #     numerator = self.group_sizes[i] * (
        #             self.alpha * (np.dot(pair[0], self.avg_states[i])[0] / np.linalg.norm(pair[0], 2)) +
        #             (1 - self.alpha) * (np.dot(pair[1], self.avg_actions[i]) / np.linalg.norm(pair[1], 2))
        #     )
        #     probability.append(numerator)
        #     denominator += numerator
        # probability /= denominator
        # # max probability
        # simulate_rewards = self.rewards[int(np.argmax(probability))]

        # calculate simulated reward in normal way
        # for idx, row in data.iterrows():
        #     state_values = row['state_float']
        #     action_values = row['action_float']
        #     numerator = self.alpha * (
        #             np.dot(pair[0], state_values)[0] / (np.linalg.norm(pair[0], 2) * np.linalg.norm(state_values, 2))
        #     ) + (1 - self.alpha) * (
        #             np.dot(pair[1], action_values)[0] / (np.linalg.norm(pair[1], 2) * np.linalg.norm(action_values, 2))
        #     )
        #     probability.append(numerator)
        #     denominator += numerator
        # probability /= denominator
        # simulate_rewards = data.iloc[int(np.argmax(probability))]['reward']

        # for k, reward in enumerate(simulate_rewards.split('|')):
        #     result += np.power(self.sigma, k) * (0 if reward == "show" else 1)
        return simulate_rewards, result
