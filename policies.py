from Robot import Robot
from Waypoint import Waypoint
import numpy as np

UP = np.array([0, -1])
DOWN = np.array([0, 1])
LEFT = np.array([-1, 0])
RIGHT = np.array([1, 0])
directions = [UP, DOWN, LEFT, RIGHT]


def nearest_waypoint(robot_position, waypoints):
    return waypoints[abs(waypoints - robot_position).sum(axis=1).argmin()]

def greedy_policy(charge_station, robot, gridworld, waypoints):
    dx, dy = nearest_waypoint(robot.position, waypoints) - waypoints
    return np.array([np.sign(dx), 0]) if abs(dx) > abs(dy) else np.array([0, np.sign(dy)])