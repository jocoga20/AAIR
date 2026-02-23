import pygame as pg
import sys
from random import randint, choices
from Robot import Robot
from Waypoint import Waypoint
from time import sleep

import numpy as np

SIZE = 50
MAX_X, MAX_Y = 16, 12
WIDTH, HEIGHT = MAX_X * SIZE, MAX_Y * SIZE

def directions(direction_vector = (0, -1)):
    x, y = direction_vector
    probs = [0.8, 0.1, 0.1]
    dirs = [direction_vector, (abs(y), abs(x)), (-abs(y), -abs(x))]
    return dirs, probs

def draw_background():
    screen.fill((255, 255, 255))

    for x in range(0, WIDTH, SIZE):
        pg.draw.line(screen, (0,0,0), (x,0), (x,HEIGHT))
    for y in range(0, HEIGHT, SIZE):
        pg.draw.line(screen, (0,0,0), (0,y), (WIDTH, y))

import policies
np.random.seed(42)
N = 5
charge_station = np.array([0, 0])
robot = Robot(0, 0, 20)
waypoints = [Waypoint(randint(0, 16), randint(0, 12)) for _ in range(N)]
visited = np.zeros(N)
state = [charge_station, robot, waypoints]

pg.init()
screen = pg.display.set_mode((WIDTH, HEIGHT))
pg.display.set_caption('AAIR')

RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

def draw(x, y, color):
    pg.draw.rect(screen, color, (x*50, y*50, 50, 50))

def draw_scene(state):
    charge_station, robot, waypoints = state
    draw_background()
    x, y = charge_station
    draw(x, y, GREEN)

    for w in waypoints:
        x, y = w.position
        draw(x, y, BLUE)
    
    x, y = robot.position
    draw(x, y, RED)

def is_allowed(p):
    x, y = p
    return 0 <= x and x < MAX_X and 0 <= y and y < MAX_Y

def allowed_directions_probability(d, robot_position):
    d1 = np.abs(d)
    d2 = -d1
    directions = [d]
    if is_allowed(robot_position + d1):
        directions.append(d1)
    if is_allowed(robot_position + d2):
        directions.append(d2)
    probabilities = [0.8, 0.2] if len(directions) < 3 else [0.8, 0.1, 0.1]
    return directions, probabilities

def choose_direction(direction, robot_position):
    d, p = allowed_directions_probability(direction, robot_position)
    return choices(d, p)

def compute_reward(state):
    charge_station, robot, waypoints = state
    reward = 0
    all_visited = True

    for w in waypoints:
        if w.visited:
            reward += 10
        else:
            all_visited = False
    
    if all_visited:
        reward += 100
    
    if robot.can_move():
        reward += 20
    
    return reward
        
    
end_episode = False
while not end_episode:
    for event in pg.event.get():
        if event.type == pg.QUIT:
            run = False
    draw_scene(state)

    reward, end_episode = compute_reward(state)
    direction = policies.greedy_policy(state)
    charge_station, robot, _ = state
    direction = choose_direction(direction, robot.position)
    
    if robot.can_move():
        robot.move(direction)
        if robot.position == charge_station:
            robot.recharge()
    else:
        end_episode = True

draw_scene(state)
sleep(2)
pg.quit()
sys.exit()