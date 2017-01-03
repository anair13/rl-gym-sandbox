import argparse
import logging
import sys

import gym
from gym import wrappers

import numpy as np
import random

STATE_DIM = 8
def observation_to_state(obs):
    x = np.array(obs)
    # x = np.array([obs[1], obs[3]])
    # return np.concatenate((np.sign(x), x * x))
    return np.concatenate((x, x * x))

class LinearQAgent(object):
    """The world's simplest Q agent!"""
    def __init__(self, action_space, logger):
        self.logger = logger
        self.action_space  = action_space
        self.theta = [np.random.randn(STATE_DIM)/10 for _ in range(action_space.n)]
        self.prev_s = None
        self.prev_a = -1

        self.i = 0
        self.alpha = 0.01
        self.gamma = 0.99
        self.epsilon = 0.2

    def Q(self, s):
        q = np.array([self.theta[i].T.dot(s) for i in range(self.action_space.n)])
        # print q
        return q

    def act(self, observation, reward, done):
        self.i += 1
        cur_s = observation_to_state(observation)
        
        # self.alpha = 1.0 / self.i
        # self.epsilon = 1.0 / self.i

        if self.prev_s is not None:
            self.theta[self.prev_a] = (1 - self.alpha) * self.theta[self.prev_a] + self.alpha * (self.gamma * np.max(self.Q(cur_s) - self.Q(self.prev_s)) + reward) * self.prev_s

        if done:
            self.prev_s = None
        else:
            self.prev_s = cur_s

        if random.random() < self.epsilon:
            q = self.Q(cur_s)
            if np.all(np.isclose(q, q[0])):
                self.prev_a = int(random.random() < 0.5) # random action
            else:
                self.prev_a = np.argmax(self.Q(cur_s))
        else:
            self.prev_a = self.action_space.sample()
        return self.prev_a