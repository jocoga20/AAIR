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
        self.font = pg.font.Font(None, 18)

        # cache: (x,y) -> (value, surface)
        self.text_cache = {}

    def draw_value(self, x, y, value):
        key = (x, y)
        cached = self.text_cache.get(key)

        # aggiorna solo se cambia valore
        if cached is None or cached[0] != value:
            surf = self.font.render(f"{value:.1f}", True, BLACK)
            self.text_cache[key] = (value, surf)
        else:
            surf = cached[1]

        # rettangolo cella
        rect = pg.Rect(x * SIZE, y * SIZE, SIZE, SIZE)

        # pulizia locale (evita sovrapposizione)
        pg.draw.rect(self.screen, WHITE, rect)
        pg.draw.rect(self.screen, BLACK, rect, 1)

        # posizione testo (leggermente centrata)
        cx = x * SIZE + SIZE * 0.2
        cy = y * SIZE + SIZE * 0.2

        self.screen.blit(surf, (cx, cy))

    def after_move(self, grid: Grid, robot: Robot):
        # ridisegna elementi dinamici sopra la griglia
        grid.draw_charge_station(self.screen)
        grid.draw_waypoints(self.screen)
        robot.draw(self.screen)

        # stato globale coerente
        _, _, _, wid = state_key(robot, grid)
        battery = robot.battery

        # disegno valori
        for x in range(MAX_X):
            for y in range(MAX_Y):
                state = (x, y, battery, wid)
                value = self.value_function.get(state)

                if value is None:
                    value = 0

                self.draw_value(x, y, value)

        pg.display.flip()
        sleep(FRAME_DRAW_TIMER)