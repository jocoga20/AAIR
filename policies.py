from Grid import Grid
from Robot import Robot
from config import *
from utils import *

def pedant_policy(grid: Grid, robot: Robot):
    if grid.all_waypoints_visited():
        return grid.direction_to_charge_station(robot.position)
    
    nearest_waypoint = grid.nearest_waypoint(robot.position)
    
    if grid.distance_between(robot.position, nearest_waypoint) + grid.distance_from_charge_station(nearest_waypoint) <= robot.battery:
        return grid.direction_to(robot.position, nearest_waypoint)
    else:
        return grid.direction_to_charge_station(robot.position)