import numpy as np
import pygame as pg
from Grid import Grid
from Robot import Robot
from config import *
from time import sleep

class NoRender:
    def __init__(self): pass
    def quitted_pygame(self): return False
    def before_move(self, grid: Grid, robot: Robot): pass
    def after_move(self, grid: Grid, robot: Robot): pass

class DrawRender(NoRender):
    def __init__(self):
        pg.init()
        self.screen = pg.display.set_mode((WIDTH, HEIGTH))
        self.screen.fill(WHITE)

        for y in range(0, HEIGTH, SIZE):
            pg.draw.line(self.screen, BLACK, (0, y), (WIDTH, y))
            
        for x in range(0, WIDTH, SIZE):
            pg.draw.line(self.screen, BLACK, (x, 0), (x, HEIGTH))
    
    def set_title(self, title):
        pg.display.set_caption(title)

    def quitted_pygame(self):
        for event in pg.event.get():
            if event.type == pg.QUIT:
                pg.quit()
                return True
        return False
    
    def before_move(self, grid: Grid, robot: Robot):
        robot.erase(self.screen)
        grid.draw_charge_station(self.screen)
        grid.draw_waypoints(self.screen)
    
    def after_move(self, grid: Grid, robot: Robot):
        robot.draw(self.screen)
        pg.display.flip()
        sleep(FRAME_DRAW_TIMER)

class DrawRenderValueFunction(DrawRender):
    def __init__(self):
        super().__init__()
    
    def before_move(self, grid: Grid, robot: Robot):
        for x in range(MAX_X):
            for y in range(MAX_Y):
                if np.all(robot.position == (x, y)):
                    robot.draw()
                    # TODO: disegna numeri value function