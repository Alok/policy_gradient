#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os

import gym
import keras
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.callbacks import EarlyStopping
from keras.layers import Activation, Dense, Input
from keras.models import Model
from pudb import set_trace

parser = argparse.ArgumentParser()
parser.add_argument('-e', '--env', type=str, default='CartPole-v1')
parser.add_argument('-n', '--iterations', type=int, default=1000)
parser.add_argument('--new', action='store_true')
parser.add_argument('--render', action='store_true')

args = parser.parse_args()

env = gym.make(args.env)

NUM_ACTIONS = env.action_space.n
STATE_SHAPE = env.observation_space.shape
STATE_SIZE = env.observation_space.shape[0]


def R(rewards, start=0, end=None) -> float:
    ''' Total discounted future rewards. '''
    return np.sum(rewards[start:end])


def init_policy(input_shape=STATE_SHAPE, depth=3):
    ''' output can be either "reward" or "action"'''

    S = Input(shape=STATE_SHAPE)

    x = Dense(32, activation='relu')(S)
    for _ in range(depth):
        x = Dense(32, activation='relu')(x)
    A = Dense(NUM_ACTIONS, activation='softmax')(x)

    return Model(inputs=S, outputs=A)


def collect_trajectory(policy, env=env):

    done = False
    states = []
    actions = []
    rewards = []
    logits = []

    state = env.reset()

    while not done:
        if args.render:
            env.render()

        logit = policy.predict(state.reshape(1, STATE_SIZE)).ravel()

        action = np.random.choice(NUM_ACTIONS, p=logit)

        state, reward, done, _ = env.step(action)

        states.append(state)
        actions.append(action)
        rewards.append(reward)
        logits.append(logit)

    # assign 0 reward to final state
    rewards.append(0.0)

    # Cast to Numpy arrays.
    states = np.array(states, dtype=np.float32)
    actions = np.array(actions, dtype=np.int)
    rewards = np.array(rewards, dtype=np.float32)
    logits = np.array(logits, dtype=np.float32)

    return states, actions, rewards, logits


if __name__ == '__main__':

    policy = keras.models.load_model(
        'policy.h5') if os.path.exists('policy.h5') and not args.new else init_policy()

    opt = keras.optimizers.Adam()
    sy_states = K.placeholder(name='states', shape=(None, STATE_SIZE))
    # loss = K.sum(K.log(policy(tf.convert_to_tensor(sy_states.reshape(-1, 4)))))

    for iter in range(args.iterations):
        states, actions, rewards, logits = collect_trajectory(policy, env)

        # Rew = np.array([sum(rewards[t:]) for t in range(len(rewards))])

        loss = sum(rewards) * K.sum(K.log(policy(sy_states)))
        updates = opt.get_updates(params=policy.weights, constraints=[], loss=loss)
        train = K.function(inputs=[sy_states], outputs=[loss], updates=updates)
        loss = train([states])
        if iter % 25 == 0:
            print(np.sum(rewards))


    # save weights
    policy.save('policy.h5')
