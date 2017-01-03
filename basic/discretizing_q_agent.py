import gym
import numpy as np
import random
from binning import safe_bin


def randarr(shape):
    N = int(np.prod(np.array(shape)))
    x = np.zeros((N))
    for i in range(N):
        x[i] = random.random()
    return np.reshape(x, shape)

BINS = 10
cartpole_bins = [(-2.5, 2.5, BINS), (-5, 5, BINS), (-.3, .3, BINS), (-4.1, 4.1, BINS)]
# cartpole_bins = [(-1, 1, BINS), (-0.5, 0.5, BINS)]
# cartpole_bins = [(-0.5, 0.5, BINS)]
def observation_to_state(obs):
    s = [safe_bin(o, *b) for o, b in zip(obs, cartpole_bins)]
    state = 0
    for d in s:
        state *= BINS
        state += d
    return state

class DiscretizingQAgent(object):
    """The world's simplest Q agent!"""
    def __init__(self, action_space, logger):
        self.logger = logger
        self.action_space  = action_space
        self.Q = np.zeros((BINS ** len(cartpole_bins), action_space.n))
        # randarr((BINS ** len(cartpole_bins), action_space.n)) * 2 - 1
        self.prev_s = -1
        self.prev_a = -1
        self.reset = True # if reset the previous state is invalid

        self.alpha = 0.1
        self.alpha_decay = 1
        self.gamma = 1
        self.epsilon = 0.5
        self.epsilon_decay = 0.99

        self.minimums = np.ones((4)) * 10000
        self.maximums = np.ones((4)) * (-10000)

    def act(self, observation, reward, done):
        """Updated Q values with received observations and makes an action."""

        self.minimums = np.minimum(self.minimums, observation)
        self.maximums = np.maximum(self.maximums, observation)
        cur_s = observation_to_state(observation)
        self.logger.debug("act %d %d" % (cur_s, reward))
        # new_obs = [observation[0], observation[2]]
        # cur_s = observation_to_state(new_obs)
        # new_obs = [observation[2]]
        # cur_s = observation_to_state(new_obs)

        Q = self.Q
        alpha = self.alpha
        gamma = self.gamma
        
        if not self.reset:
            old_Q = Q[self.prev_s, self.prev_a]
            Q[self.prev_s, self.prev_a] = (1 - alpha) * Q[self.prev_s, self.prev_a] + alpha * (reward + gamma * np.max(Q[cur_s, :]))
            self.logger.debug("updating Q(%d,%d): %f -> %f" % (self.prev_s, self.prev_a, old_Q, Q[self.prev_s, self.prev_a]))

        self.prev_s = cur_s

        self.reset = False
        if done:
            self.reset = True
        
        if random.random() > self.epsilon:
            q = self.Q[cur_s, :]
            # if np.all(np.isclose(q, q[0])): # Q values are all the same
            #     self.prev_a = random.randint(0, self.action_space.n - 1)
            #     self.logger.debug("action: %d (random Q)" % self.prev_a)
            # else:
            self.prev_a = np.argmax(q)
            self.logger.debug("action: %d (Q)" % self.prev_a)
        else:
            self.prev_a = random.randint(0, self.action_space.n - 1)
            self.logger.debug("action: %d (random)" % self.prev_a)

        self.epsilon *= self.epsilon_decay
        self.alpha *= self.alpha_decay

        return self.prev_a