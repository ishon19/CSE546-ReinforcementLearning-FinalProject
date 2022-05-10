
from copy import copy
from random import random

import numpy as np


# constants
MU = 0
SIGMA = 0.2
THETA = 0.15

class Noise:
    def __init__(self, size, seed):
        random.seed(seed), np.random.seed(seed)
        self.size = size, self.mu = MU * np.ones(size)
        self.theta = THETA, self.sigma = SIGMA
        self.reset()

    def reset(self):
        self.state = copy.copy(self.mu)

    def sample(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.size)
        self.state = x + dx
        return self.state