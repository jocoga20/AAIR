import pygame as pg
import sys

from Robot import Robot
from Waypoint import Waypoint

import numpy as np

WIDTH, HEIGHT = 800, 600
SIZE = 50

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

N = 10
charge_station = np.array([0, 0])
robot = Robot(0, 0, 10)
waypoints = np.random.randint(low=5, high=12, size=(N, 2)) * 50

pg.init()
screen = pg.display.set_mode((WIDTH, HEIGHT))
pg.display.set_caption('AAIR')

clock = pg.time.Clock()

run = True

RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

from time import sleep

def draw(x, y, color):
    pg.draw.rect(screen, color, (x, y, 50, 50))

while run:
    for event in pg.event.get():
        if event.type == pg.QUIT:
            run = False

    draw_background()

    x, y = charge_station
    draw(x, y, GREEN)

    for x, y in waypoints:
        draw(x, y, BLUE)
    
    x, y = robot.position
    draw(x, y, RED)
    
    u, hovering, wpi = policies.greedy_policy(robot, waypoints)
    robot.move(u * 50)

    if hovering:
        waypoints = np.delete(waypoints, wpi, axis=0)
        print(waypoints.shape)

    pg.display.flip()
    sleep(0.5)
    

pg.quit()
sys.exit()