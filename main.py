#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse

import gym
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn.utils
from torch import Tensor as T
from torch import nn, stack
from torch.autograd import Variable as V
from torch.nn.functional import relu, softplus
from torch.optim import Adam

parser = argparse.ArgumentParser()
parser.add_argument('-e', '--env', type=str, default='Pendulum-v0')
parser.add_argument('-n', '--iterations', type=int, default=2_000)
parser.add_argument('-d', '--discount', type=float, default=.99)
parser.add_argument('--new', action='store_true')
parser.add_argument('--render', '-r', action='store_true')

args = parser.parse_args()

DISCOUNT = args.discount
ITERS = args.iterations

env = gym.make(args.env)

STATE_SHAPE = env.observation_space.shape[0] if len(
    env.observation_space.shape
) == 1 else env.observation_space.shape

ACTION_SHAPE = env.action_space.shape[0] if len(
    env.action_space.shape
) == 1 else env.action_space.shape


def W(x=np.random.rand(STATE_SHAPE)) -> V:
    '''Wrap array into Variable. '''
    if isinstance(x, list):
        var = V(T(x))
    elif isinstance(x, np.ndarray):
        var = V(torch.from_numpy(x))
    elif isinstance(x, (float, int)):
        var = V(T([x]))
    return var


class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        S = STATE_SHAPE
        H = 128  # Hidden size
        A = ACTION_SHAPE
        self.l1 = nn.Linear(S, H)
        self.l2 = nn.Linear(H, H)
        self.l3 = nn.Linear(H, H)

        self.mean_head = nn.Linear(H, A)
        self.variance_head = nn.Linear(H, A)

    def forward(self, s: V) -> (V, V):
        '''Output mean and variance of a Gaussian.'''
        s = s.view(1, STATE_SHAPE).float()
        s = self.l1(s)
        s = relu(s)
        s = self.l2(s)
        s = relu(s)
        s = self.l3(s)
        s = relu(s)

        mean = self.mean_head(s)

        # Apply a softplus to ensure variance nonnegative.
        variance = softplus(self.variance_head(s))

        return mean, variance


def sample(mean: V, variance: V) -> T:
    '''Sample an action. Since we pass it to gym, no need for Variable output.'''
    std = variance.sqrt()
    action = torch.normal(mean, std).data

    return action


def log_pdf(a: T, mean: V, variance: V) -> V:
    '''Get log probability density of an action to try and avoid overflow.'''
    # To avoid some of the downsides of writing this myself, manually unroll the log of a product to avoid over/underflow
    exp_term = (-(V(a) - mean).pow(2) / (2 * variance))
    log_coeff = -((2 * np.pi * variance).sqrt()).log()
    return (log_coeff + exp_term).view(1)


def G(rewards, start=0, end=None):
    return sum(rewards[start:end])


def bprop(opt, rewards, log_probs) -> V:
    '''Statefully perform optimization.'''
    discounted_rewards = [pow(DISCOUNT, i) * r for i, r in enumerate(rewards)]
    acc_returns = [G(discounted_rewards, t) for t in range(len(discounted_rewards))]
    GAE = W(acc_returns)

    log_probs = stack(log_probs)

    loss = -(GAE @ log_probs) / len(rewards)

    opt.zero_grad()
    loss.backward()
    opt.step()
    return loss


if __name__ == '__main__':

    policy = Policy()
    opt = Adam(policy.parameters())

    for i in range(ITERS):
        if args.render:
            env.render()

        rewards = []
        log_probs = []

        done = False

        s = env.reset()

        while not done:
            mean, variance = policy(W(s))

            a = sample(mean, variance)
            log_prob = log_pdf(a, mean, variance)

            s, r, done, _ = env.step(a.numpy()[0])

            log_probs.append(log_prob)
            rewards.append(r)

        # TODO discount
        Adv = sum(V(T(rewards)))
        logits = stack(logits)
        loss = -sum(Adv.expand_as(logits) * logits)
        N = len(rewards)
        loss = loss / N

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm(policy.parameters(), 40)
        opt.step()
        if i % 1000 == 0:
            print(int(Adv.data.numpy()))
