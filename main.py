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
    probabilities = [0.8, 0.2] if len(directions) == 2 else [0.8, 0.1, 0.1]
    return random.choices(directions, probabilities)[0]

myseed = 42
random.seed(myseed)
np.random.seed(myseed)
grid = Grid.random_generate(N=5)
robot = Robot(x=0, y=0, full_battery=60)
score = 0

vf = ValueFunction(MAX_X, MAX_Y)

pg.init()
screen = pg.display.set_mode((WIDTH, HEIGTH))
pg.display.set_caption('AAIR')

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

while grid.episode_continues:
    for event in pg.event.get():
        if event.type == pg.QUIT:
            pg.quit()
            break
    
    screen.fill(WHITE)
    
    for y in range(SIZE, HEIGTH, SIZE):
        pg.draw.line(screen, BLACK, (0, y), (WIDTH, y))
    
    for x in range(SIZE, WIDTH, SIZE):
        pg.draw.line(screen, BLACK, (x, 0), (x, HEIGTH))

    x, y = grid.charge_station
    pg.draw.rect(screen, GREEN, (x*SIZE, y*SIZE, SIZE, SIZE))

    for w in grid.waypoints:
        x, y = w
        pg.draw.rect(screen, BLUE, (x*SIZE, y*SIZE, SIZE, SIZE))

    x, y = robot.position
    pg.draw.rect(screen, RED, (x*SIZE, y*SIZE, SIZE, SIZE))
    direction = policies.smart_policy(robot, grid)
#    direction = choose_direction(direction, robot.position)
    robot.move(direction)

    reward = grid.compute_reward(robot)
    x, y = robot.position
    vf.update(x, y, reward)
    score += reward
    
#    print(f'{score} ({f'+{reward}' if reward > 0 else reward})')

    print(vf.status)
    pg.display.flip()
    sleep(0.1)
