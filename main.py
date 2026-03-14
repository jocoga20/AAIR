from time import sleep

import numpy as np
import random
from Grid import Grid
from ValueFunction import ValueFunction
import policies
from Robot import Robot
from config import *
import pygame as pg

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

def state_key(robot: Robot, grid: Grid):
    x, y = robot.position
    x, y = x.item(), y.item()

    return x, y, robot.battery, grid.waypoints_status

def direction_value(direction: np.array, robot: Robot, grid: Grid, value_function: ValueFunction):
    robot.move(direction)
    value = value_function.get(state_key(robot, grid))
    robot.move(-direction)
    robot.battery += 2
    return value

def best_action(robot: Robot, grid: Grid, actions: list[np.array], value_function: ValueFunction):
    return [direction_value(d, robot, grid, value_function) for d in actions]

def experiment_draw(seed):
    random.seed(seed)
    np.random.seed(seed)

    charge_station = np.zeros(2, 'int32')

    N = 5
    grid = Grid.random_generate(N=N, charge_station=charge_station)

    robot = Robot(x=0, y=0, full_battery=100)
    old_state_key = state_key(robot, grid)

    discount_factor = 0.9
    score = 0

    vf = ValueFunction(step_size=0.9, discount_factor=discount_factor)

    pg.init()
    screen = pg.display.set_mode((WIDTH, HEIGTH))
    pg.display.set_caption(f'AAIR seed {seed}')

    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    RED = (255, 0, 0)
    GREEN = (0, 255, 0)
    BLUE = (0, 0, 255)

    screen.fill(WHITE)

    for y in range(0, HEIGTH, SIZE):
        pg.draw.line(screen, BLACK, (0, y), (WIDTH, y))
        
    for x in range(0, WIDTH, SIZE):
        pg.draw.line(screen, BLACK, (x, 0), (x, HEIGTH))

    while grid.episode_continues:
        for event in pg.event.get():
            if event.type == pg.QUIT:
                pg.quit()
                break

        x, y = grid.charge_station
        pg.draw.rect(screen, GREEN, (x*SIZE+1, y*SIZE+1, SIZE-1, SIZE-1))

        for w in grid.waypoints:
            x, y = w
            pg.draw.rect(screen, BLUE, (x*SIZE+1, y*SIZE+1, SIZE-1, SIZE-1))

        # clear old robot position before redrawing
        x, y = robot.position
        pg.draw.rect(screen, WHITE, (x*SIZE+1, y*SIZE+1, SIZE-1, SIZE-1))

        direction = policies.greedy_policy(robot, grid)
        direction = choose_direction(direction, robot.position)
        robot.move(direction)

        x, y = robot.position
        pg.draw.rect(screen, RED, (x*SIZE+1, y*SIZE+1, SIZE-1, SIZE-1))

        reward = grid.compute_reward(robot)
        score += discount_factor * reward
        new_state_key = state_key(robot, grid)
        vf.update(old_state_key, new_state_key, reward)
        old_state_key = new_state_key
        pg.display.flip()
        sleep(1)
    return score

def experiment(seed):
    random.seed(seed)
    np.random.seed(seed)

    charge_station = np.zeros(2, 'int32')

    N = 5
    grid = Grid.random_generate(N=N, charge_station=charge_station)

    robot = Robot(x=0, y=0, full_battery=100)
    old_state_key = state_key(robot, grid)

    discount_factor = 0.9
    score = 0

    vf = ValueFunction(step_size=0.9, discount_factor=discount_factor)

    while grid.episode_continues:
        direction = policies.greedy_policy(robot, grid)
        direction = choose_direction(direction, robot.position)
        robot.move(direction)

        reward = grid.compute_reward(robot)
        score += discount_factor * reward
        new_state_key = state_key(robot, grid)
        vf.update(old_state_key, new_state_key, reward)
        old_state_key = new_state_key
    return score

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

results = [experiment(seed) for seed in range(1_000)]

for bins in [100, 200, 400, 1000]:
    plot_hist(results, bins)