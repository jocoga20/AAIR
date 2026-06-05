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
    def paused(self): pass
    def check_events(self): return self

class DrawRender(NoRender):
    def __init__(self, frame_draw_time:float=0.5):
        self.pause = False
        self.quitted = False
        self.frame_draw_time = frame_draw_time
        pg.init()
        self.screen = pg.display.set_mode((WIDTH, HEIGTH))
        self.screen.fill(WHITE)
        for x in range(0, WIDTH, SIZE):
            pg.draw.line(self.screen, BLACK, (x, 0), (x, HEIGTH))
        
        for y in range(0, HEIGTH, SIZE):
            pg.draw.line(self.screen, BLACK, (0, y), (WIDTH, y))
    
    def set_title(self, title):
        pg.display.set_caption(title)

    def quitted_pygame(self):
        return self.quitted

    def paused(self):
        return self.pause

    def check_events(self):
        for event in pg.event.get():
            if event.type == pg.QUIT:
                self.quitted = True
            elif event.type == pg.KEYDOWN:
                if event.key == pg.K_p:
                    self.pause = not self.pause
                    print(f'Paused: {self.pause}')
                elif event.key == pg.K_q:
                    self.quitted = True

        return self
    
    def before_move(self, grid: Grid, robot: Robot):
        robot.erase(self.screen)
        grid.draw_charge_station(self.screen)
        grid.draw_waypoints(self.screen)
    
    def after_move(self, grid: Grid, robot: Robot):
        robot.draw(self.screen)
        pg.display.flip()
        sleep(self.frame_draw_time)
        

class TimePlotRender(DrawRender):
    def __init__(self, value_function: ValueFunction, frame_draw_time: float=0.5):
        super().__init__(frame_draw_time)
        self.value_function = value_function
        self.font = pg.font.Font(None, 20)
        self.text_cache = {}

    def draw_value(self, x, y, value, bgcolor):
        key = (x, y)
        cached = self.text_cache.get(key)

        if cached is None or cached[0] != value:
            fmt_value = f"{value:.1e}"
            fmt_value = fmt_value\
                .replace("e+0", "e")\
                .replace("e+", "e")\
                .replace("e-0", "e-")
            
            surface = self.font.render(fmt_value, True, BLACK)
            self.text_cache[key] = (value, surface)
        else:
            surface = cached[1]
            
        rect = pg.Rect(x * SIZE+1, y * SIZE+1, SIZE-1, SIZE-1)

        pg.draw.rect(self.screen, bgcolor, rect)

        cx = x * SIZE + SIZE * 0.1
        cy = y * SIZE + SIZE * 0.4

        self.screen.blit(surface, (cx, cy))

    def before_move(self, grid, robot):
        robot.erase(self.screen)

    def after_move(self, grid: Grid, robot: Robot):
        _, _, battery, wid = state_key(robot, grid)
        for x in range(MAX_X):
            for y in range(MAX_Y):
                bgcolor = WHITE
                if robot.position[0] == x and robot.position[1] == y:
                    bgcolor = RED
                elif grid.charge_station[0] == x and grid.charge_station[1] == y:
                    bgcolor = GREEN
                elif grid.has_waypoint_at(x, y):
                    bgcolor = BLUE

                value = self.value_function.get((x, y, battery, wid))
                self.draw_value(x, y, value, bgcolor)
        pg.display.flip()
        sleep(self.frame_draw_time)