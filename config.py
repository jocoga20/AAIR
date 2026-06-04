COMPLETE_REWARD = 1000
WAYPOINT_REWARD = 100
CHARGE_REWARD = 0
EMPTY_REWARD = -1
FAIL_REWARD = -200

MAX_X = 20          # how many cells in a row
MAX_Y = 20          # how many cells in a column

SIZE = 50           # size of a cell (remember 1px for the border)
WIDTH, HEIGTH = MAX_X * SIZE, MAX_Y * SIZE

FULL_BATTERY = (MAX_X + MAX_Y) * 2      # max level of battery when the robot charges at the station or starts the episode
# this formula makes sure the robot can travel to the furthest point of the grid world and then come back

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)


def step_size_default_rule(t):
    return 1 / (t + 1)          # satisfies the Robbins-Monro condition (c1 = c2 = 1)