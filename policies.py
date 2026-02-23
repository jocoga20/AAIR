from Robot import Robot
from Waypoint import Waypoint
import numpy as np

UP = np.array([0, -1])
DOWN = np.array([0, 1])
LEFT = np.array([-1, 0])
RIGHT = np.array([1, 0])
directions = [UP, DOWN, LEFT, RIGHT]

def dist(x, y):
    return np.linalg.norm(x-y)

def nearest_waypoint(position: np.array, waypoints: np.array):
    d = np.linalg.norm(waypoints - position, axis=1)
    i = d.argmin()
    return i

def greedy_policy(state):
    charge_station, robot, waypoints = state
    wpi = nearest_waypoint(robot.position, waypoints)
    dx, dy = waypoints[wpi] - robot.position

    if abs(dx) > abs(dy):
        dir = np.array([np.sign(dx), 0])
    else:
        dir = np.array([0, np.sign(dy)])
    
    return dir