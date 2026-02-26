import numpy as np


class ValueFunction:
    def __init__(self, MAX_X, MAX_Y, beta = 0.5):
        self.status = np.zeros((MAX_X, MAX_Y))
        self.beta = beta
    
    def update(self, x, y, reward):
        self.status[x, y] = self.beta * (reward - self.status[x, y]) + (1 - self.beta) * (self.status[x, y])
