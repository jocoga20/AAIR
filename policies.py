from Grid import Grid
from Robot import Robot



def greedy_policy(robot: Robot, grid: Grid):
    """
    Just go to the nearest waypoint. It will surely die if battery is not sufficient.
    """
    if grid.all_waypoints_visited():
        return grid.direction_to_charge_station(robot.position)
    return grid.direction_to_nearest_new_waypoint(robot.position)

def smart_policy(robot: Robot, grid: Grid):
    """
    Go to the nearest waypoint, but go charge if battery is not capable enough
    """
    if grid.all_waypoints_visited():
        return grid.direction_to_charge_station(robot.position)
    
    min_dist = grid.distances_from_new_waypoints(robot.position).min()
    if robot.battery < min_dist:
        return grid.direction_to_charge_station(robot.position)
    return grid.direction_to_nearest_new_waypoint(robot.position)

def strict_smart_policy(robot: Robot, grid: Grid):
    """
    More conservative than the latter. Goes to waypoint if (ideally) it can also come back
    """
    if grid.all_waypoints_visited():
        return grid.direction_to_charge_station(robot.position)
    
    min_dist = grid.distances_from_new_waypoints(robot.position).min()
    if robot.battery < min_dist * 2:
        return grid.direction_to_charge_station(robot.position)
    return grid.direction_to_nearest_new_waypoint(robot.position)

def one_by_one_policy(robot: Robot, grid: Grid):
    """
    Charge station if task completed. Waypoint is able to go to w and then charge
    """
    if grid.all_waypoints_visited():
        return grid.direction_to_charge_station(robot.position)
    
    wp = grid.nearest_new_waypoint(robot.position)
    d_wp = grid.distance_between(robot.position, wp)
    d_cs = grid.distance_from_charge_station(wp)
    if robot.battery >= d_wp + d_cs:
        return grid.direction_to(arrival=wp, start=robot.position)
    return grid.direction_to_charge_station(robot.position)
