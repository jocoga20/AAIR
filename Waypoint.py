import numpy as np
import pygame as pg

class Waypoint:
    def __init__(self, position = np.array):
        self.position = position
        self.size = 50
    
    def draw(self, screen):
        x, y = self.position
        pg.draw.rect(screen, (0,0,255), (x, y, self.size, self.size))