from Grid import Grid
from Robot import Robot

def state_key(robot: Robot, grid: Grid):
    x, y = robot.position
    x, y = x.item(), y.item()

    return x, y, robot.battery, grid.waypoints_status