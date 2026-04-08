from Grid import Grid
from Robot import Robot
from config import *
from utils import *

def greedy_policy(grid: Grid, robot: Robot):
    if grid.all_waypoints_visited():
        return grid.direction_to_charge_station(robot.position)
    return grid.direction_to_nearest_waypoint(robot.position)

def pedant_policy(grid: Grid, robot: Robot):
    if grid.all_waypoints_visited():
        return grid.direction_to_charge_station(robot.position)
    
    nw = grid.nearest_waypoint(robot.position)
    if grid.distance_between(robot.position, nw) + grid.distance_from_charge_station(nw) <= robot.battery:
        return grid.direction_to(robot.position, nw)
    else:
        return grid.direction_to_charge_station(robot.position)