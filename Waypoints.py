import numpy as np

class Waypoint:
    def __init__(self, x, y):
        self.position = np.array([x, y])
        self.visited = False