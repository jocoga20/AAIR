import numpy as np
from random import choices
from Grid import Grid
import policies
from Robot import Robot
from config import *

def is_allowed(position):
    x, y = position
    return 0 <= x and x < MAX_X and 0 <= y and y < MAX_Y

def choose_direction(direction, robot_position):
    dx, dy = direction
    d1 = np.array([dy, dx])
    d2 = -d1
    directions = [direction] + [d for d in [d1, d2] if is_allowed(d + robot_position)]
    probabilities = [0.8, 0.2] if len(directions) == 2 else [0.8, 0.1, 0.1]
    return choices(directions, probabilities)[0]

g = Grid.random_generate(N=5)

while g.episode_continues:
    