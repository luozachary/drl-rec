#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created by luozhenyu on 2018/11/26
"""
import random
import copy
import numpy as np


class Simulator(object):
    def __init__(self, alpha=0.5, sigma=0.9):
        # self.data = data
        self.alpha = alpha
        self.sigma = sigma
        self.init_state = self.reset()
        self.current_state = self.init_state
        self.rewards, self.group_sizes, self.avg_states, self.avg_actions = self.avg_group()

    def reset(self):
        init_state = np.random.rand(1, 12, 30)
        return init_state

    def step(self, action):
        # pair_state = copy.deepcopy(self.current_state).reshape((1, 360))
        # pair_action = copy.deepcopy(action).reshape((1, 120))
        # simulate_rewards, result = self.simulate_reward((pair_state, pair_action))
        simulate_rewards, result = self.simulate_reward((self.current_state.reshape((1, 360)),
                                                         action.reshape((1, 120))))

        print("reward type and value: {0} / {1}".format(type(result), result))

        for i, r in enumerate(simulate_rewards):
            if r > 0:
                # self.current_state.append(action[i])
                tmp = np.append(self.current_state[0], action[i].reshape((1, 30)), axis=0)
                tmp = np.delete(tmp, 0, axis=0)
                self.current_state = tmp[np.newaxis, :]
        return result, self.current_state

    def avg_group(self):
        """calculate average state/action value for each group."""
        # rewards = list()
        # avg_states = list()
        # avg_actions = list()
        # group_sizes = list()
        # for reward, group in self.data.groupby(['reward']):
        #     n_size = group.size()
        #     state_values = group['state'].values
        #     action_values = group['action'].values
        #     avg_states.append(
        #         np.sum(state_values / np.linalg.norm(state_values, 2, axis=1)[:, np.newaxis], axis=0) / n_size
        #     )
        #     avg_actions.append(
        #         np.sum(action_values / np.linalg.norm(action_values, 2, axis=1)[:, np.newaxis], axis=0) / n_size
        #     )
        #     group_sizes.append(n_size)
        #     rewards.append(reward)
        # return rewards, group_sizes, avg_states, avg_actions

        # test data
        nums = [0, 1, 5]
        rewards = list()
        reward = list()
        helper(rewards, reward, nums)
        group_sizes = [100] * 81
        avg_states = np.random.rand(81, 360)
        avg_actions = np.random.rand(81, 120)
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
        result = 0.
        for i, reward in enumerate(self.rewards):
            numerator = self.group_sizes[i] * (
                    self.alpha * (np.dot(pair[0], self.avg_states[i])[0] / np.linalg.norm(pair[0], 2)) +
                    (1 - self.alpha) * (np.dot(pair[1], self.avg_actions[i]) / np.linalg.norm(pair[1], 2))
            )
            probability.append(numerator)
            denominator += numerator
        probability /= denominator
        # 最大相似
        # simulate_rewards = np.sum(np.asarray(self.rewards) * np.asarray(probability)[:, np.newaxis], axis=0)
        simulate_rewards = np.asarray(self.rewards[int(np.argmax(probability))])
        print("****************simulator shape****************:", simulate_rewards.shape)

        for k, reward in enumerate(simulate_rewards):
            result += np.power(self.sigma, k) * reward
        return simulate_rewards, result


def helper(rewards, reward, nums):
    """需要删除，生成测试reward数据"""
    if len(reward) == 4:
        tmp = reward.copy()
        rewards.append(tmp)
        return
    for i, val in enumerate(nums):
        reward.append(val)
        helper(rewards, reward, nums)
        reward.pop()
