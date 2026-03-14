from time import sleep

import numpy as np
import random
from Grid import Grid
from ValueFunction import ValueFunction
import policies
from Robot import Robot
from config import *
import pygame as pg
from utils import *

def is_allowed(position):
    x, y = position
    return 0 <= x and x < MAX_X and 0 <= y and y < MAX_Y

def filter_allowed(directions, position):
    return [d for d in directions if is_allowed(position + d)]

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
    robot = Robot(x=x, y=y, full_battery=100)
    grid = Grid.random_generate(N=nwaypoints, charge_station=charge_station)
    return grid, robot

def init_graphics(seed):
    pg.init()
    screen = pg.display.set_mode((WIDTH, HEIGTH))
    pg.display.set_caption(f'AAIR seed {seed}')
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

def experiment_draw(seed, vf, nwaypoints = N_WAYPOINTS, charge_station_position = np.zeros(2, 'int32')):
    set_seed(seed)
    grid, robot = setup(nwaypoints, charge_station_position)

    s0 = state_key(robot, grid)
    score = 0
    screen = init_graphics(seed)

    while grid.episode_continues:
        if pygame_quit():
            break

        grid.draw_charge_station(screen)
        grid.draw_waypoints(screen)
        robot.erase(screen)

        allowed_directions = filter_allowed(DIRECTIONS, robot.position)
        direction = policies.epsilon_greedy(allowed_directions, robot, grid, vf)
        # direction = choose_direction(direction, robot.position)
        robot.move(direction)
        robot.draw(screen)

        reward = grid.compute_reward(robot)
        score += DISCOUNT_FACTOR * reward
        s1 = state_key(robot, grid)
        vf.update(s0, s1, reward)
        s0 = s1

        pg.display.flip()
        sleep(FRAME_DRAW_TIMER)

    return score

def experiment(seed, vf, nwaypoints = N_WAYPOINTS, charge_station_position = np.zeros(2, 'int32')):
    set_seed(seed)
    grid, robot = setup(nwaypoints, charge_station_position)

    s0 = state_key(robot, grid)
    score = 0

    while grid.episode_continues:
        allowed_directions = filter_allowed(DIRECTIONS, robot.position)
        direction = policies.epsilon_greedy(allowed_directions, robot, grid, vf)
        # direction = choose_direction(direction, robot.position)
        robot.move(direction)

        reward = grid.compute_reward(robot)
        score += DISCOUNT_FACTOR * reward
        s1 = state_key(robot, grid)
        vf.update(s0, s1, reward)
        s0 = s1

    return score

import matplotlib
matplotlib.use("TkAgg")   # oppure QtAgg

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

vf = ValueFunction(step_size=0.9, discount_factor=DISCOUNT_FACTOR)

x_vals = []
y_vals = []

plt.ion()

fig, ax = plt.subplots()
line, = ax.plot(x_vals, y_vals, marker='o')

ax.set_xlabel("iteration")
ax.set_ylabel("score")
ax.set_title("Score per episode")

def update_plot(x, y):
    x_vals.append(x)
    y_vals.append(y)

    line.set_xdata(x_vals)
    line.set_ydata(y_vals)
    ax.relim()
    ax.autoscale_view()
    fig.canvas.draw()
    fig.canvas.flush_events()


plt.show(block=False)   # fondamentale

eps_exp = 2/3
eps_std = 1 - eps_exp
nepisode_exp = 20

EPSILON = eps_exp
for it in range(20):
    score = experiment(42, vf)
    update_plot(it, score)

plt.savefig('img.png')

EPSILON = eps_std
for it in range(nepisode_exp, 800 + nepisode_exp):
    score = experiment(42, vf)
    update_plot(it, score)
    

for it in range(800 + nepisode_exp, 800 + nepisode_exp + 10):
    score = experiment_draw(42, vf)
    update_plot(it, score)

plt.savefig('img.png')