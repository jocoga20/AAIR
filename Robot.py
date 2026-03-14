import numpy as np
from config import *
import pygame as pg

class Robot:
    def __init__(self, x, y, full_battery = 20):
        self.position = np.array([x, y])
        self.battery = full_battery
        self.full_battery = full_battery
    
    def move(self, direction):
        self.position += direction
        self.battery -= 1
    
    def can_move(self):
        return self.battery > 0

    def recharge(self):
        self.battery = self.full_battery
    
    def erase(self, screen):
        x, y = self.position
        pg.draw.rect(screen, WHITE, (x*SIZE+1, y*SIZE+1, SIZE-1, SIZE-1))
    
    def draw(self, screen):
        x, y = self.position
        pg.draw.rect(screen, (255 * self.battery / self.full_battery, 0, 0), (x*SIZE+1, y*SIZE+1, SIZE-1, SIZE-1))