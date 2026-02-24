import numpy as np
from random import choices
import policies

class Robot:
    def __init__(self, x, y, full_battery = 20):
        self.position = np.array([x, y])
        self.battery = full_battery
        self.full_battery = full_battery
    
    def move(self, direction):
        self.position += direction
        self.battery -= 1
    
    def can_move(self):
        return self.battery > 0

    def recharge(self):
        self.battery = self.full_battery

WAYPOINT_REWARD = 10
CHARGE_REWARD = 0
EMPTY_REWARD = -1
MAX_X = MAX_Y = 20

def setup_gridworld(N=5):
    charge_station = np.array([0, 0])
    np.random.seed(42)
    gridworld = -np.ones((20, 20), dtype='int8')
    gridworld[tuple(charge_station)] = 0
    waypoints = np.random.randint(20, size=(N, 2))

    for w in waypoints:
        gridworld[tuple(w)] = WAYPOINT_REWARD

    return charge_station, gridworld, waypoints

def is_allowed(position):
    x, y = position
    return 0 <= x and x < MAX_X and 0 <= y and y < MAX_Y

def choose_direction(direction, robot_position):
    dx, dy = direction
    d1 = np.array([dy, dx])
    d2 = -d1
    directions = [direction] + [d for d in [d1, d2] if is_allowed(d + robot_position)]
    probabilities = [0.8, 0.2] if len(directions) == 2 else [0.8, 0.1, 0.1]
    return choices(directions, probabilities)

charge_station, gridworld, waypoints = setup_gridworld()

robot = Robot(0, 0)
print(gridworld)
score = 0
running_episode = True


while running_episode:
    direction = policies.greedy_policy(charge_station, robot, gridworld, waypoints)
    direction = choose_direction(direction, robot.position)

    if robot.can_move():
        robot.move(direction)
    else:
        break

    p = tuple(robot.position)

    if gridworld[p] == WAYPOINT_REWARD:
        score += WAYPOINT_REWARD
        gridworld[p] = EMPTY_REWARD
        np.delete()
    elif p == charge_station:
        robot.recharge()
        score += CHARGE_REWARD
    else:
        score += gridworld[p]