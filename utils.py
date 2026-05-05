from Grid import Grid
from Robot import Robot
from config import *

def state_key(robot: Robot, grid: Grid):
    """
    This function sums up the state of the ambient (agent included) in a key to access and manipulate the value function
    """
    x, y = robot.position
    x, y = x.item(), y.item()

    return (x, y, robot.battery, grid.waypoints_status)