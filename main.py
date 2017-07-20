#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import random
import subprocess
import sys
import typing
from collections import namedtuple
from pathlib import Path
from typing import Any, NamedTuple, Tuple

import gym
import keras
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.callbacks import EarlyStopping
from keras.layers import Activation, Dense
from pudb import set_trace

parser = argparse.ArgumentParser()
parser.add_argument('-a', '--algorithm', type=str, default='pg')
parser.add_argument('-e', '--env', type=str, default='CartPole-v0')
parser.add_argument('-n', '--iterations', type=int, default=10)
parser.add_argument('--new', action='store_true')

args = parser.parse_args()

env = gym.make(args.env)

NUM_ACTIONS = env.action_space.n
STATE_SHAPE = env.observation_space.shape

################# TYPES #####################
Real = float
parameter = Any
timestep = int

Action = int  # 0 for left, 1 for right
Actions = typing.Sequence[Action]  # 0 for left, 1 for right
State = typing.Sequence  # (4,) array
States = typing.Sequence  # (4,) array
Reward = float
Rewards = typing.Sequence[Reward]


class Trajectory(NamedTuple):
    states: Any = None
    actions: Any = None
    rewards: Any = None


Network = typing.Callable
Policy = typing.Callable[[State], Action]
Baseline = typing.Callable[[State], Reward]

#############################################

# TODO add gamma as default arg = .95


def J(w: parameter) -> Real:
    '''Objective function of parameters'''
    raise NotImplementedError


def init_network(output: str, input_shape=STATE_SHAPE, depth=3) -> Network:
    ''' output can be either "reward" or "action"'''
    model = keras.models.Sequential()
    # input layer
    model.add(Dense(32, activation='relu', input_shape=input_shape))

    # middle layers
    for _ in range(depth):
        model.add(Dense(32, activation='relu'))

    # output layer
    # TODO does this work for a baseline?
    if output == 'reward':
        model.add(Dense(1, activation='linear'))
    elif output == 'action':
        model.add(Dense(1, activation='sigmoid'))
    else:
        print('Output must be reward or action', file=sys.stderr)

    model.compile(loss=keras.losses.mse, optimizer=keras.optimizers.Adam(), metrics=['accuracy'])

    return model


def initialize_policy(input_shape=STATE_SHAPE, depth=3) -> Policy:
    return init_network(input_shape=input_shape, depth=depth, output='action')


def initialize_baseline(input_shape=STATE_SHAPE, depth=3) -> Baseline:
    return init_network(input_shape=input_shape, depth=depth, output='reward')


def R(rewards, start: timestep=0, end: timestep=None) -> Real:
    ''' Total discounted future rewards. '''
    return np.sum(rewards[start:end])


# def Adv(rewards: Rewards, t: timestep) -> Real:
#     return R(rewards, t) - b(rewards, t)


def collect_trajectory(pi: Policy, env) -> Trajectory:

    # to avoid python's list append behavior
    trajectory = Trajectory()
    if trajectory.states is None:
        trajectory = trajectory._replace(states=[])
    if trajectory.actions is None:
        trajectory = trajectory._replace(actions=[])
    if trajectory.rewards is None:
        trajectory = trajectory._replace(rewards=[])

    done = False

    state = env.reset()
    trajectory.states.append(state)

    while not done:
        # env.render()
        # need to wrap in np.array([]) for `predict` to work
        action = 1 if pi.predict(np.array([state])) > random.random() else 0
        state, reward, done, _ = env.step(action)
        trajectory.states.append(state)
        trajectory.actions.append(action)
        trajectory.rewards.append(reward)
    # Cast elements of tuple to Numpy arrays.
    # assign 0 reward to final state
    trajectory.rewards.append(0.0)
    trajectory = Trajectory(
        np.array(trajectory.states),
        np.array(trajectory.actions),
        np.array(trajectory.rewards), )

    return trajectory


if __name__ == '__main__':
    pi = keras.models.load_model('policy.h5') if os.path.exists('policy.h5') else initialize_policy()
    b = keras.models.load_model('baseline.h5') if os.path.exists('baseline.h5') else initialize_baseline()

    for _ in range(500):
        states, actions, rewards = collect_trajectory(pi, env)

        Rew = np.array([sum(rewards[t:]) for t in range(len(rewards))])

        b.fit(x=states, y=Rew, epochs=2, validation_split=0.1, callbacks=[EarlyStopping()])
    # save weights
    b.save('baseline.h5')
    pi.save('policy.h5')
