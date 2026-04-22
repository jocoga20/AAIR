from Grid import Grid
from Robot import Robot
from config import *
import random
import numpy as np
import pygame as pg

def state_key(robot: Robot, grid: Grid):
    """
    This function sums up the state of the ambient (agent included) in a key to access and manipulate the value function
    """
    x, y = robot.position
    x, y = x.item(), y.item()

    return x, y, robot.battery, grid.waypoints_status

def is_allowed(position):
    x, y = position
    return 0 <= x and x < MAX_X and 0 <= y and y < MAX_Y

def choose_direction(direction, robot_position, pmax = 0.9):
    """
    Chooses between the main direction and one or more orthogonal direction(s).
    A direction is allowed if it does not brings the robot outside the grid world.
    The main direction is always chosen with probability **pmax**.
    If there is one allowed orthogonal direction, probability of choosing it is **1 - pmax**.
    If there are two allowed orthogonal directions, probability of choosing each of them is **(1 - pmax)/2**.
    """
    dx, dy = direction
    d1 = np.array([dy, dx])
    d2 = -d1
    directions = [direction] + [d for d in [d1, d2] if is_allowed(d + robot_position)]
    probabilities = [pmax, 1-pmax] if len(directions) == 2 else [pmax, (1-pmax)/2, (1-pmax)/2]
    return random.choices(directions, probabilities)[0]

def set_seed(seed):
    """
    Sets all the seeds needed to reproduce the experiment.
    """
    random.seed(seed)
    np.random.seed(seed)

def init_ambient(num_waypoints, charge_station):
    """
    This function returns a Grid and a Robot instance.
    Modifiy this function to customize the beginning of the experiment.
    """
    x, y = charge_station
    robot = Robot(x=x, y=y, full_battery=(MAX_X + MAX_Y)*2)
    grid = Grid.random_generate(num_waypoints, charge_station)
    return grid, robot

def init_graphics(title):
    """
    This function only runs when in drawing mode. 
    """
    pg.init()
    screen = pg.display.set_mode((WIDTH, HEIGTH))
    pg.display.set_caption(title)
    screen.fill(WHITE)

    for y in range(0, HEIGTH, SIZE):
        pg.draw.line(screen, BLACK, (0, y), (WIDTH, y))
        
    for x in range(0, WIDTH, SIZE):
        pg.draw.line(screen, BLACK, (x, 0), (x, HEIGTH))
    return screen

def quitted_pygame():
    for event in pg.event.get():
        if event.type == pg.QUIT:
            pg.quit()
            return True
    return False