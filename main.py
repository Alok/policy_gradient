#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import math

import gym
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn.utils
from pudb import set_trace
from torch import Tensor as T
from torch import nn, stack
from torch.autograd import Variable as V
from torch.nn.functional import relu, softplus
from torch.optim import Adam

parser = argparse.ArgumentParser()
parser.add_argument('-e', '--env', type=str, default='Pendulum-v0')
parser.add_argument('-n', '--iterations', type=int, default=10000)
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
DISCOUNT = .99

pi = V(T([math.pi]))


def W(s: np.ndarray = np.random.rand(3)) -> V:
    '''Wrap array into Variable '''
    return V(torch.from_numpy(s))


class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        S = STATE_SHAPE
        H = 128  # Hidden size
        A = ACTION_SHAPE
        self.l1 = nn.Linear(S, H)
        self.l2 = nn.Linear(H, H)

        self.l3, self.l3_ = nn.Linear(H, A), nn.Linear(H, 1)

    def forward(self, s: V):
        s = s.view(1, 3).float()
        s = relu(self.l1(s))
        s = relu(self.l2(s))

        mean = self.l3(s)
        # Variance must be > 0, so apply a softplus
        variance = softplus(self.l3_(s))
        return mean, variance


def sample(mean: V, variance: V) -> T:
    '''Sample an action'''

    std = variance.sqrt()
    # Sample from standard normal distribution to scale into arbitrary Gaussian.
    gaussian_noise = V(torch.randn(mean.size()))

    # `action` SHOULD be a Tensor, not a Variable.
    action = (mean + std * gaussian_noise).data

    return torch.clamp(
        action,
        min=float(env.action_space.low),
        max=float(env.action_space.high),
    )


def pdf(a: T, mean: V, variance: V) -> V:
    '''Get probability density of an action'''
    exp_term = (-(V(a) - mean).pow(2) / (2 * variance)).exp()
    coeff = 1 / ((2 * variance * pi.expand_as(variance)).sqrt())
    return coeff * exp_term


def log_pdf(a: T, mean: V, variance: V) -> V:
    '''Get log probability density of an action to try and avoid overflow.'''
    # Don't apply exponent since we take a log anyway.
    exp_term = (-(V(a) - mean).pow(2) / (2 * variance))
    # XXX. Experiment: To avoid some of the downsides of writing this myself, manually unroll the log to avoid over/underflow
    # coeff = 1 / ((2 * variance * pi.expand_as(variance)).sqrt())
    log_coeff = -((2 * variance * pi.expand_as(variance)).sqrt()).log()
    # return (coeff.log() + exp_term).view(1)
    return (log_coeff + exp_term).view(1)


if __name__ == '__main__':

    behavior_policy = Policy()

    for _ in range(args.iterations):
        states, actions, rewards = collect_trajectory(behavior_policy)

        returns = G(rewards)

    # TODO save weights
