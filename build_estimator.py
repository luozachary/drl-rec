#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created by luozhenyu on 2018/11/26
"""
import argparse
import numpy as np
import tensorflow as tf
import pprint as pp
from replay_buffer import RelayBuffer
from simulator import Simulator
from pre_process_data import recall_data
from util.logger import logger


class Actor(object):
    """policy function approximator"""
    def __init__(self, sess, s_dim, a_dim, batch_size, output_size, weights_len, tau, learning_rate, scope="actor"):
        self.sess = sess
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.batch_size = batch_size
        self.output_size = output_size
        self.weights_len = weights_len
        self.tau = tau
        self.learning_rate = learning_rate
        self.scope = scope

        with tf.variable_scope(self.scope):
            # estimator actor network
            self.state, self.action_weights = self._build_net("estimator_actor")
            self.network_params = tf.trainable_variables()

            # target actor network
            self.target_state, self.target_action_weights = self._build_net("target_actor")
            self.target_network_params = tf.trainable_variables()[len(self.network_params):]

            # operator for periodically updating target network with estimator network weights
            self.update_target_network_params = [
                self.target_network_params[i].assign(
                    tf.multiply(self.network_params[i], self.tau) +
                    tf.multiply(self.target_network_params[i], 1 - self.tau)
                ) for i in range(len(self.target_network_params))
            ]
            self.hard_update_target_network_params = [
                self.target_network_params[i].assign(
                    self.network_params[i]
                ) for i in range(len(self.target_network_params))
            ]

            self.a_gradient = tf.placeholder(tf.float32, [None, self.a_dim])
            self.params_gradients = list(
                map(
                    lambda x: tf.div(x, self.batch_size),
                    tf.gradients(tf.reshape(self.action_weights, [self.batch_size, self.a_dim]),
                                 self.network_params, -self.a_gradient)
                )
            )
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate).apply_gradients(
                zip(self.params_gradients, self.network_params)
            )
            self.num_trainable_vars = len(self.network_params) + len(self.target_network_params)

    def _build_net(self, scope):
        """build the tensorflow graph"""
        with tf.variable_scope(scope):
            state = tf.placeholder(tf.float32, [None, self.s_dim], "state")
            state_ = tf.reshape(state, [-1, self.weights_len, int(self.s_dim / self.weights_len)])
            cell = tf.nn.rnn_cell.GRUCell(self.output_size,
                                          activation=tf.nn.relu,
                                          kernel_initializer=tf.initializers.random_normal(),
                                          bias_initializer=tf.zeros_initializer())
            outputs, final_state = tf.nn.dynamic_rnn(cell, state_, dtype=tf.float32, time_major=False)
        return state, outputs

    def train(self, state, a_gradient):
        self.sess.run(self.optimizer, feed_dict={self.state: state, self.a_gradient: a_gradient})

    def predict(self, state):
        return self.sess.run(self.action_weights, feed_dict={self.state: state})

    def predict_target(self, state):
        return self.sess.run(self.target_action_weights, feed_dict={self.target_state: state})

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)

    def hard_update_target_network(self):
        self.sess.run(self.hard_update_target_network_params)

    def get_num_trainable_vars(self):
        return self.num_trainable_vars


class Critic(object):
    """value function approximator"""
    def __init__(self, sess, s_dim, a_dim, num_actor_vars, gamma, tau, learning_rate, scope="critic"):
        self.sess = sess
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.num_actor_vars = num_actor_vars
        self.gamma = gamma
        self.tau = tau
        self.learning_rate = learning_rate
        self.scope = scope

        with tf.variable_scope(self.scope):
            # estimator critic network
            self.state, self.action, self.q_value = self._build_net("estimator_critic")
            self.network_params = tf.trainable_variables()[self.num_actor_vars:]

            # target critic network
            self.target_state, self.target_action, self.target_q_value = self._build_net("target_critic")
            self.target_network_params = tf.trainable_variables()[(len(self.network_params) + self.num_actor_vars):]

            # operator for periodically updating target network with estimator network weights
            self.update_target_network_params = [
                self.target_network_params[i].assign(
                    tf.multiply(self.network_params[i], self.tau) +
                    tf.multiply(self.target_network_params[i], 1 - self.tau)
                ) for i in range(len(self.target_network_params))
            ]
            self.hard_update_target_network_params = [
                self.target_network_params[i].assgin(
                    self.network_params[i]
                ) for i in range(len(self.target_network_params))
            ]

            self.predicted_q_value = tf.placeholder(tf.float32, [None, 1])
            self.loss = tf.reduce_mean(tf.squared_difference(self.predicted_q_value, self.q_value))
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
            self.a_gradient = tf.gradients(self.q_value, self.action)

    def _build_net(self, scope):
        with tf.variable_scope(scope):
            state = tf.placeholder(tf.float32, [None, self.s_dim], "state")
            action = tf.placeholder(tf.float32, [None, self.a_dim], "action")
            inputs = tf.concat([state, action], axis=-1)
            layer1 = tf.layers.Dense(64, activation=tf.nn.relu)(inputs)
            layer2 = tf.layers.Dense(64, activation=tf.nn.relu)(layer1)
            q_value = tf.layers.Dense(1)(layer2)
            return state, action, q_value

    def train(self, state, action, predicted_q_value):
        return self.sess.run([self.q_value, self.loss, self.optimizer], feed_dict={
            self.state: state,
            self.action: action,
            self.predicted_q_value: predicted_q_value
        })

    def predict(self, state, action):
        return self.sess.run(self.q_value, feed_dict={self.state: state, self.action: action})

    def predict_target(self, state, action):
        return self.sess.run(self.target_q_value, feed_dict={self.target_state: state, self.target_action: action})

    def action_gradients(self, state, action):
        return self.sess.run(self.a_gradient, feed_dict={self.state: state, self.action: action})

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)

    def hard_update_target_network(self):
        self.sess.run(self.hard_update_target_network_params)


class OUNoise:
    """noise for action"""
    def __init__(self, a_dim, mu=0, theta=0.5, sigma=0.2):
        self.a_dim = a_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.a_dim) * self.mu
        self.reset()

    def reset(self):
        self.state = np.ones(self.a_dim) * self.mu

    def noise(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.rand(len(x))
        self.state = x + dx
        return self.state


def gene_actions(item_space, weight_batch, action_len):
    """use output of actor network to calculate action list
    Args:
        item_space: recall items
        weight_batch: actor network outputs
        action_len: length of recommendation list

    Returns:
        recommendation list
    """
    action_batch = list()
    for weight in weight_batch:
        action = list()
        space = item_space.copy()
        for i in range(action_len):
            score = np.dot(space, weight[i])
            idx = np.argmax(score)
            action.append(space[idx])
            space = np.delete(space, idx, axis=0)
        action_batch.append(action)
    return np.asarray(action_batch)


def build_summaries():
    episode_reward = tf.Variable(0.)
    tf.summary.scalar("reward", episode_reward)
    episode_max_q = tf.Variable(0.)
    tf.summary.scalar("max_q_value", episode_max_q)
    critic_loss = tf.Variable(0.)
    tf.summary.scalar("critic_loss", critic_loss)

    summary_vars = [episode_reward, episode_max_q, critic_loss]
    summary_ops = tf.summary.merge_all()
    return summary_ops, summary_vars


def learn_from_batch(replay_buffer, batch_size, actor, critic, item_space, action_len, s_dim, a_dim):
    if replay_buffer.size() < batch_size:
        pass
    samples = replay_buffer.sample_batch(batch_size)
    state_batch = np.asarray([_[0] for _ in samples])
    action_batch = np.asarray([_[1] for _ in samples])
    reward_batch = np.asarray([_[2] for _ in samples])
    n_state_batch = np.asarray([_[3] for _ in samples])

    # calculate predicted q value
    action_weights = actor.predict_target(state_batch)
    n_action_batch = gene_actions(item_space, action_weights, action_len)
    target_q_batch = critic.predict_target(n_state_batch.reshape((-1, s_dim)), n_action_batch.reshape((-1, a_dim)))
    y_batch = []
    for i in range(batch_size):
        y_batch.append(reward_batch[i] + critic.gamma * target_q_batch[i])

    # train critic
    q_value, critic_loss, _ = critic.train(state_batch, action_batch, np.reshape(y_batch, (batch_size, 1)))
    # train actor
    action_weight_batch_for_gradients = actor.predict(state_batch)
    action_batch_for_gradients = gene_actions(item_space, action_weight_batch_for_gradients, action_len)
    a_gradient_batch = critic.action_gradients(state_batch, action_batch_for_gradients.reshape((-1, a_dim)))
    actor.train(state_batch, a_gradient_batch[0])

    # update target networks
    actor.update_target_network()
    critic.update_target_network()

    return np.amax(q_value), critic_loss


def train(sess, env, actor, critic, exploration_noise, s_dim, a_dim, args):
    # set up summary operators
    summary_ops, summary_vars = build_summaries()
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter(args['summary_dir'], sess.graph)

    # initialize target network weights
    actor.hard_update_target_network()
    critic.hard_update_target_network()

    # initialize replay memory
    replay_buffer = RelayBuffer(int(args['buffer_size']))

    for i in range(int(args['max_episodes'])):
        ep_reward = 0.
        ep_q_value = 0.
        loss = 0.
        item_space = recall_data
        state = env.reset()
        # update average parameters every 1000 episodes
        if (i + 1) % 10 == 0:
            env.rewards, env.group_sizes, env.avg_states, env.avg_actions = env.avg_group()
        for j in range(args['max_episodes_len']):
            weight = actor.predict(np.reshape(state, [1, s_dim])) + exploration_noise.noise().reshape(
                (1, int(args['action_item_num']), int(a_dim / int(args['action_item_num'])))
            )
            action = gene_actions(item_space, weight, int(args['action_item_num']))
            reward, n_state = env.step(action[0])
            replay_buffer.add(list(state.reshape((s_dim,))),
                              list(action.reshape((a_dim,))),
                              [reward],
                              list(n_state.reshape((s_dim,))))
            ep_reward += reward
            ep_q_value_, critic_loss = learn_from_batch(replay_buffer, args['batch_size'], actor, critic, item_space,
                                                        args['action_item_num'], s_dim, a_dim)
            ep_q_value += ep_q_value_
            loss += critic_loss
            state = n_state
            if j % 50 == 0:
                logger.info("=========={0} episode of {1} round: {2} reward=========".format(i, j, reward))
            summary_str = sess.run(summary_ops, feed_dict={summary_vars[0]: ep_reward,
                                                           summary_vars[1]: ep_q_value,
                                                           summary_vars[2]: loss})
            writer.add_summary(summary_str, i)

    writer.close()


def main(args):
    # init memory data
    # data = load_data()
    with tf.Session() as sess:
        # simulated environment
        env = Simulator()
        s_dim = int(args['embedding']) * int(args['state_item_num'])
        a_dim = int(args['embedding']) * int(args['action_item_num'])

        actor = Actor(sess, s_dim, a_dim,
                      int(args['batch_size']), int(args['embedding']),
                      int(args['action_item_num']), float(args['tau']),
                      float(args['actor_lr']))

        critic = Critic(sess, s_dim, a_dim,
                        actor.get_num_trainable_vars(), float(args['gamma']),
                        float(args['tau']), float(args['critic_lr']))

        exploration_noise = OUNoise(a_dim)

        train(sess, env, actor, critic, exploration_noise, s_dim, a_dim, args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="provide arguments for DDPG agent")

    # agent parameters
    parser.add_argument("--embedding", help="dimension of item embedding", default=30)
    parser.add_argument("--state_item_num", help="click history list length for user", default=12)
    parser.add_argument("--action_item_num", help="length of the recommendation item list", default=4)
    parser.add_argument("--actor_lr", help="actor network learning rate", default=0.0001)
    parser.add_argument("--critic_lr", help="critic network learning rate", default=0.001)
    parser.add_argument("--gamma", help="discount factor for critic updates", default=0.99)
    parser.add_argument("--tau", help="soft target update parameter", default=0.001)
    parser.add_argument("--buffer_size", help="max size of the replay buffer", default=1000000)
    parser.add_argument("--batch_size", help="size of minibatch for minbatch-SGD", default=64)

    # run parameters
    parser.add_argument("--max_episodes", help="max num of episodes to do while training", default=50000)
    parser.add_argument("--max_episodes_len", help="max length of 1 episode", default=100)
    parser.add_argument("--summary_dir", help="directory for storing tensorboard info", default='./results')

    args_ = vars(parser.parse_args())
    logger.info(pp.pformat(args_))

    main(args_)
