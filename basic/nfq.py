import gym
import numpy as np
import random

import q_model

class NFQAgent(object):
    """The world's simplest Q agent!"""
    def __init__(self, action_space, logger):
        self.logger = logger
        self.action_space  = action_space
        self.hyperparams = q_model.hyperparams()
        self.Q = q_model.QModel(self.hyperparams)

        self.prev_s = -1
        self.prev_a = -1
        self.reset = True # if reset the previous state is invalid
        self.episode = 0

        self.gamma = 0.99
        self.epsilon = 0.1
        self.epsilon_decay = 1

        self.data = []

    def act(self, observation, reward, done):
        """Updated Q values with received observations and makes an action."""
        cur_s = observation
        # self.logger.debug("act %d %d" % (cur_s, reward))

        q = self.Q.f(observation)
        self.logger.debug("observation %s q %s" % (observation, q))
        
        if not self.reset:
            if done:
                t = reward
            else:
                t = reward + self.gamma * np.max(q)
            d = (self.prev_s, self.prev_a, t)
            self.data.append(d)
            # self.logger.debug("updating Q(%d,%d): %f -> %f" % (self.prev_s, self.prev_a, old_Q, Q[self.prev_s, self.prev_a]))

        self.prev_s = cur_s

        self.reset = False
        if done:
            self.reset = True
            self.episode += 1
            if self.episode % self.hyperparams["trainEpisodes"] == 0:
                self.Q.train(self.data)
                self.data = []
        
        if random.random() > self.epsilon and self.episode > self.hyperparams["trainEpisodes"]:
            self.prev_a = np.argmax(q)
            self.logger.debug("action: %d (Q)" % self.prev_a)
        else:
            self.prev_a = random.randint(0, self.action_space.n - 1)
            self.logger.debug("action: %d (random)" % self.prev_a)

        self.epsilon *= self.epsilon_decay

        return self.prev_a