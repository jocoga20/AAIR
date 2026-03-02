import numpy as np


class ValueFunction:
    def __init__(self, default_value = 0, beta = 0.5):
        self.sparse_dict = {}
        self.beta = beta
        self.default_value = default_value

    def update(self, x, y, battery, waypoints_status, reward):
        key = x, y, battery, waypoints_status
        v0 = self.get(x, y, battery, waypoints_status)
        alpha = 1
        v1 = v0 + self.beta * (reward - v0)
        self.sparse_dict[key] = reward
    
    def get(self, x, y, battery, waypoints_status):
        x, y = x.item(), y.item()
        key = x, y, battery, waypoints_status
        return self.sparse_dict.get(key, self.default_value)
    
    def __str__(self):
        return self.status_list