from Grid import Grid
from Robot import Robot

def greedy_policy(robot: Robot, grid: Grid):
    return grid.direction_to_nearest_new_waypoint(robot.position)

def smart_policy(robot: Robot, grid: Grid):
    min_dist = grid.distances_from_new_waypoints(robot.position).min()
    if robot.battery < min_dist * 2:
        return grid.direction_to_charge_station(robot.position)
    return grid.direction_to_nearest_new_waypoint(robot.position)