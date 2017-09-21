#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import math

import gym
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.autograd import Variable
from torch.nn.functional import relu

parser = argparse.ArgumentParser()
parser.add_argument('-e', '--env', type=str, default='Pendulum-v0')
parser.add_argument('-n', '--iterations', type=int, default=2)
parser.add_argument('--new', action='store_true')
parser.add_argument('--render', '-r', action='store_true')

args = parser.parse_args()

ITERS = args.iterations
env = gym.make(args.env)

STATE_SHAPE = env.observation_space.shape[0]
ACTION_SHAPE = env.action_space.shape[0]
# States are Numpy arrays of length 3 with elements in the range [1,1,8]
STATE_RANGE = env.observation_space.high
# Actions are Numpy arrays of length 1 with elements in the (negative and positive) range [,2]
ACTION_RANGE = env.action_space.high

################# TYPES #####################
timestep = int

Reward = float

#############################################

# TODO add gamma as default arg = .95
DISCOUNT = .99

pi = Variable(torch.FloatTensor([math.pi]))

# TODO policy should return a probability (prob vector over discrete actions or just a prob)


def discount(rewards):
    # return Tensor([pow(DISCOUNT, i) for i in range(len(rewards))])
    return np.array([pow(DISCOUNT, i) for i in range(len(rewards))])


def G(rewards, start: timestep = 0, end: timestep = None) -> Reward:
    '''Total discounted future rewards.'''
    return sum(np.array(rewards[start:end]) * discount(rewards[start:end]))


def collect_trajectory(policy, *, render=args.render):
    '''Run through an episode by following a policy and collect data.'''

    # to avoid python's list append behavior
    states = []
    actions = []
    rewards = []

    done = False

    s = Tensor(env.reset())
    states.append(s) # if s is a scalar, this will just return random numbers

    while not done:
        if render:
            env.render()
        # TODO add action picking
        s = Variable(s)
        a = behavior_policy.select_action(s)[0].data.numpy()
        s, r, done, _ = env.step(a)
        s = Tensor(s)
        states.append(s)
        actions.append(a)
        rewards.append(r)

    # Final state is considered to have reward 0 for transitioning forever.
    rewards.append(0)

    return states, actions, rewards


def gaussian(x, mean, var):
    a = (-(x - mean).pow(2) / (2 * var)).exp()
    b = 1 / ((2 * var * pi.expand_as(var)).sqrt())
    return a * b


class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        S = STATE_SHAPE
        H = 128  # Hidden size
        A = ACTION_SHAPE
        self.l1 = nn.Linear(S, H)
        self.l2 = nn.Linear(H, H)

        self.l3, self.l3_ = nn.Linear(H, A), nn.Linear(H, 1)

    def forward(self, s):
        s = s.view(1, 3).float()
        s = self.l1(s)
        s = self.l2(s)

        mean, var = self.l3(s), self.l3_(s)

        return mean, var

    def select_action(self, s):

        mean, var = self.forward(s)

        action = mean + var.sqrt() * Variable(torch.randn(mean.size()))
        action = torch.normal(mean, var.sqrt())
        prob = gaussian(action, mean, var)

        log_prob = prob.log()

        return action, log_prob


if __name__ == '__main__':

    behavior_policy = Policy()

    for _ in range(args.iterations):
        states, actions, rewards = collect_trajectory(behavior_policy)

        returns = G(rewards)

    # TODO save weights
