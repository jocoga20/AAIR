import numpy as np
import pygame as pg

class Robot:
    def __init__(self, x = 0, y = 0, full_battery = 10):
        self.position = np.array([x, y])
        self.battery = full_battery
        self.full_battery = full_battery
        self.size = 50
    
    def draw(self, screen):
        x, y = self.position
        pg.draw.rect(screen, (255,0,0), (x, y, self.size, self.size))
    
    def move(self, direction):
        x, y = direction
        self.position[0] += x
        self.position[1] += y