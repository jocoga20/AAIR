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

myseed = 42
random.seed(myseed)
np.random.seed(myseed)
charge_station = np.zeros(2, 'int32')
N = 5
grid = Grid.random_generate(N=N, charge_station=charge_station)
full_battery=2*grid.distance_to_furthest_waypoint(charge_station)
robot = Robot(x=0, y=0, full_battery=full_battery)
score = 0

vf = ValueFunction()

pg.init()
screen = pg.display.set_mode((WIDTH, HEIGTH))
pg.display.set_caption('AAIR')

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

    x, y = robot.position
    pg.draw.rect(screen, WHITE, (x*SIZE+1, y*SIZE+1, SIZE-1, SIZE-1))
    direction = policies.one_by_one_policy(robot, grid)
    direction = choose_direction(direction, robot.position)
    robot.move(direction)
    x, y = robot.position
    pg.draw.rect(screen, RED, (x*SIZE+1, y*SIZE+1, SIZE-1, SIZE-1))

    reward = grid.compute_reward(robot)
    x, y = robot.position
    vf.update(x, y, robot.battery, grid.waypoints_status, reward)
    score += reward
    
#    print(f'{score} ({f'+{reward}' if reward > 0 else reward})')

    #print(vf.status)
#    print('w', grid.distances_from_new_waypoints(robot.position).min())
#    print('c', grid.distance_from_charge_station(robot.position))
    pg.display.flip()
#    sleep(0.1)

print(vf.status_list)