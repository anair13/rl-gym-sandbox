import gym
import numpy as np
import random

import q_model

class RandomLinearAgent(object):
    """The world's simplest linear agent!"""
    def __init__(self, action_space, logger, params=None):
        self.logger = logger
        self.action_space  = action_space
        if params is None:
            self.params = np.random.rand(4) * 2 - 1
            print self.params
            # self.params = np.array([1, 0, 1, 0])
        else:
            self.params = params

    def act(self, observation, reward, done):
        """Updated Q values with received observations and makes an action."""
        action = 0 if observation.dot(self.params) < 0 else 1
        return action