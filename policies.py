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

def dist_from_nearest_waypoint(position: np.array, waypoints: np.array):
    d = np.linalg.norm(waypoints - position, axis=1)
    i = d.argmin()
    return i, d[i]

def greedy_policy(robot: Robot, waypoints: np.array):
    min_dist = float('inf')
    min_dir = None
    min_wpi = -1

    for direction in directions:
        wpi, md = dist_from_nearest_waypoint(robot.position + direction, waypoints)
        
        if md < min_dist:
            min_wpi = wpi
            min_dist = md
            min_dir = direction
    
    hovering = min_dist == 0

    return min_dir, hovering, min_wpi