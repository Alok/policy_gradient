#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import typing
from typing import Any, NamedTuple, Tuple

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
parser.add_argument('-n', '--iterations', type=int, default=10)
parser.add_argument('--new', action='store_true')
parser.add_argument('--render', action='store_true')

args = parser.parse_args()

env = gym.make(args.env)

NUM_ACTIONS = env.action_space.n
STATE_SHAPE = env.observation_space.shape

################# TYPES #####################
Real = float

Action = int  # 0 for left, 1 for right
State = typing.Sequence  # (4,) array
Probability = float


class Trajectory(NamedTuple):
    states: Any = None
    actions: Any = None
    rewards: Any = None
    action_probs: Any = None


Network = typing.Callable
Policy = typing.Callable[[State, Action], Probability]

#############################################


def R(rewards, start=0, end=None) -> Real:
    ''' Total discounted future rewards. '''
    return np.sum(rewards[start:end])


def collect_trajectory(policy: Policy, env=env) -> Trajectory:

    # to avoid python's list append behavior
    trajectory = Trajectory()
    if trajectory.states is None:
        trajectory = trajectory._replace(states=[])
    if trajectory.actions is None:
        trajectory = trajectory._replace(actions=[])
    if trajectory.action_probs is None:
        trajectory = trajectory._replace(action_probs=[])
    if trajectory.rewards is None:
        trajectory = trajectory._replace(rewards=[])

    done = False

    state = env.reset()
    trajectory.states.append(state)

    while not done:
        if args.render:
            env.render()
        # need to wrap in np.array([]) for `predict` to work
        # `predict` returns 2D array with single 1D array that we need to extract
        action_probs = policy(tf.convert_to_tensor(state.astype('float32').reshape(1,STATE_SHAPE[0]))).eval()
        action = np.random.choice(np.arange(NUM_ACTIONS), p=action_probs[0])

        action = np.random.choice(np.arange(NUM_ACTIONS), p=action_probs)
        state, reward, done, _ = env.step(action)

        trajectory.states.append(state)
        trajectory.actions.append(action)
        trajectory.action_probs.append(action_probs[action])
        trajectory.rewards.append(reward)

    # Cast elements of tuple to Numpy arrays.
    # assign 0 reward to final state
    trajectory.rewards.append(0.0)
    trajectory = Trajectory(
        np.array(trajectory.states),
        np.array(trajectory.actions),
        np.array(trajectory.action_probs),
        np.array(trajectory.rewards), )

    return trajectory


def init_policy(input_shape=STATE_SHAPE, depth=3) -> Network:
    ''' output can be either "reward" or "action"'''

    S = Input(shape=STATE_SHAPE)

    x = Dense(32, activation='relu')(S)
    for _ in range(depth):
        x = Dense(32, activation='relu')(x)
    A = Dense(NUM_ACTIONS, activation='softmax')(x)

    return Model(inputs=S, outputs=A)


opt = keras.optimizers.Adam()

# updates = opt.get_updates([],[],K.)

if __name__ == '__main__':
    policy = keras.models.load_model(
        'policy.h5') if os.path.exists('policy.h5') and not args.new else init_policy()

    x = K.placeholder(name='x', shape=(None, 4))
    y = K.placeholder(name='y', shape=(None, 2))
    # loss = K.sum(K.log(policy.predict()), axis=None)

    for _ in range(args.iterations):
        states, actions, action_probs, rewards = collect_trajectory(policy, env)

        Rew = np.array([sum(rewards[t:]) for t in range(len(rewards))])

    # save weights
    policy.save('policy.h5')
