import numpy as np
from random import choices
import policies
from Robot import Robot
from config import *

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
    return choices(directions, probabilities)[0]

def compute_reward(charge_station, robot: Robot, gridworld, waypoints):
    p = tuple(robot.position)
    reward = 0
    end_episode = False

    if gridworld[p] == WAYPOINT_REWARD:
        gridworld[p] = EMPTY_REWARD
        try:
            i = np.where(np.all(waypoints == robot.position))
        except ValueError:
            print('wp', waypoints.shape)
        reward = WAYPOINT_REWARD
        waypoints = np.delete(waypoints, i)
    elif np.all(p == charge_station):
        robot.recharge()
        reward = CHARGE_REWARD
    else:
        reward = EMPTY_REWARD
    
    if len(waypoints) == 0:
        end_episode = True
        reward += COMPLETE_REWARD
    else:
        if not robot.can_move():
            end_episode = True
            reward += FAIL_REWARD

    return reward, end_episode, waypoints


charge_station, gridworld, waypoints = setup_gridworld()

robot = Robot(0, 0)
score = 0
end_episode = False

while not end_episode:
    direction = policies.greedy_policy(charge_station, robot, gridworld, waypoints)
    direction = choose_direction(direction, robot.position)

    robot.move(direction)

    reward, end_episode, waypoints = compute_reward(charge_station, robot, gridworld, waypoints)
    score += reward
    print(f'{score} ({reward})')