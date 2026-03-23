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
            grid.draw_charge_station(screen)
            grid.draw_waypoints(screen)
            robot.erase(screen)

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

vf = ValueFunction(step_size_lambda=STEP_SIZE_RULE, reward_discount=REWARD_DISCOUNT)

x_vals = []
y_vals = []

plt.ion()

fig, ax = plt.subplots()
line, = ax.plot(x_vals, y_vals, marker='o')

ax.set_xlabel("iteration")
ax.set_ylabel("score")
ax.set_title("Score per episode")
drawing = False

def update_plot(x, y):
    x_vals.append(x)
    y_vals.append(y)

    line.set_xdata(x_vals)
    line.set_ydata(y_vals)
    ax.relim()
    ax.autoscale_view()
    fig.canvas.draw()
    fig.canvas.flush_events()

def plot_scores(success, fails):
    scount = len(success)
    fcount = len(fails)
    plt.scatter(list(range(scount)), success, c='red', label=f'success ({scount})')
    plt.scatter(list(range(scount, scount+fcount)), fails, c='blue', label=f'fail ({fcount})')
    plt.ylabel('scores')
    plt.xlabel('-')
    plt.legend(loc='lower left')
    plt.savefig('scores.png')

plt.show(block=False)
success_scores = []
fail_scores = []

it = 0
def cycle_for(times):
    global it
    for _ in range(times):
        score, completed = experiment(42+it, vf, 'AAIR')
        if completed:
            success_scores.append(score)
        else:
            fail_scores.append(score)
        it += 1

cycle_for(10_000)
plot_scores(success_scores, fail_scores)