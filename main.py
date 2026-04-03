from time import sleep

import numpy as np
import random
from Grid import Grid
from ValueFunction import ValueFunction
from ValueFunctionLambda import ValueFunctionLambda
import policies
from Robot import Robot
from config import *
import pygame as pg
from utils import *

def is_allowed(position):
    x, y = position
    return 0 <= x and x < MAX_X and 0 <= y and y < MAX_Y

def choose_direction(direction, robot_position):
    dx, dy = direction
    d1 = np.array([dy, dx])
    d2 = -d1
    directions = [direction] + [d for d in [d1, d2] if is_allowed(d + robot_position)]
    pmax = 0.9
    probabilities = [pmax, 1-pmax] if len(directions) == 2 else [pmax, (1-pmax)/2, (1-pmax)/2]
    return random.choices(directions, probabilities)[0]

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

def setup(nwaypoints, charge_station):
    x, y = charge_station
    robot = Robot(x=x, y=y, full_battery=(MAX_X + MAX_Y)*2)
    grid = Grid.random_generate(N=nwaypoints, charge_station=charge_station)
    return grid, robot

def init_graphics(seed, title):
    pg.init()
    screen = pg.display.set_mode((WIDTH, HEIGTH))
    pg.display.set_caption(f'AAIR seed {seed} - {title}')
    screen.fill(WHITE)

    for y in range(0, HEIGTH, SIZE):
        pg.draw.line(screen, BLACK, (0, y), (WIDTH, y))
        
    for x in range(0, WIDTH, SIZE):
        pg.draw.line(screen, BLACK, (x, 0), (x, HEIGTH))
    return screen

def pygame_quit():
    for event in pg.event.get():
        if event.type == pg.QUIT:
            pg.quit()
            return True
    return False

def experiment(seed, vf, title, nwaypoints = N_WAYPOINTS, charge_station_position = np.zeros(2, 'int32')):
    set_seed(seed)
    grid, robot = setup(nwaypoints, charge_station_position)

    s0 = state_key(robot, grid)
    score = 0
    if drawing:
        screen = init_graphics(seed, title)

    while grid.episode_continues:
        if drawing and pygame_quit():
            break

        if drawing:
            robot.erase(screen)
            grid.draw_charge_station(screen)
            grid.draw_waypoints(screen)

        direction = policies.pedant_policy(grid, robot)
        direction = choose_direction(direction, robot.position)
        robot.move(direction)

        if drawing:
            robot.draw(screen)

        reward = grid.compute_reward(robot)
        score += REWARD_DISCOUNT * reward
        s1 = state_key(robot, grid)
        vf.update(s0, s1, reward)
        s0 = s1

        if drawing:
            pg.display.flip()
            sleep(FRAME_DRAW_TIMER)

    return score, grid.mission_complete

import matplotlib.pyplot as plt

def plot_hist(values, bins=20):
    values = np.array(values)
    
    plt.figure()
    plt.hist(values, bins=bins)
    plt.axvline(values.mean(), linestyle="dashed", label=f"mean = {values.mean():.3f}")
    
    plt.legend()
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.title("Histogram")
    plt.savefig(f'hist{bins}.png')

vf = ValueFunctionLambda(step_size_lambda=STEP_SIZE_RULE, reward_discount=REWARD_DISCOUNT)
drawing = False
import numpy as np

def plot_vf(vf: ValueFunction):
    xs = np.array(list(vf.value_dict.values()))
    xs.sort()
    plt.scatter(list(range(len(xs))), xs)
    plt.show()

for it in range(100):
    experiment(seed=42 + it, vf=vf, title='AAIR')

plot_vf(vf)