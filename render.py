import numpy as np
import pygame as pg
from Grid import Grid
from Robot import Robot
from ValueFunction import ValueFunction
from config import *
from time import sleep

from utils import state_key

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
    def __init__(self, value_function: ValueFunction):
        super().__init__()
        self.value_function = value_function
        self.font = pg.font.Font(None, 16)
        self.texts = [[self.font.render(str(0), True, BLACK, WHITE)  for y in range(MAX_Y)] for x in range(MAX_X)]

    def index_to_pos(self, i):
        return SIZE * (i + 0.5)
    
    def draw_value(self, x, y, value):
        cx = self.index_to_pos(x)
        cy = self.index_to_pos(y)
        self.texts[x][y] = self.font.render(str(value), True, BLACK, WHITE)
        self.screen.blit(self.texts[x][y], (cx, cy))

    def after_move(self, grid: Grid, robot: Robot):
        robot.draw(self.screen)
        for x in range(MAX_X):
            for y in range(MAX_Y):
                wid = state_key(robot, grid)[-1]
                value = self.value_function.get((x,y,robot.battery,wid))
                
                self.draw_value(x, y, value)
        pg.display.flip()
        sleep(FRAME_DRAW_TIMER)