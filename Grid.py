import numpy as np
from Robot import Robot
from config import *
import pygame as pg

class Grid:
    def __init__(self, waypoints: np.array, charge_station = np.zeros(2, 'int32')):
        self.waypoints = waypoints
        self.waypoints_status = 0
        self.charge_station = charge_station
        self.episode_continues = True
        self.mission_complete = False
        self.waypoints_counter = waypoints.shape[0]

    @staticmethod
    def __plane_pos_to_index(pos: np.array):
        x, y = pos
        return y * MAX_Y + x

    @staticmethod
    def random_generate(num_waypoints: int, charge_station = np.zeros(2, 'int32')):
        i = Grid.__plane_pos_to_index(charge_station)
        coords = np.arange(0, MAX_X * MAX_Y)
        coords = np.random.choice(coords[coords != i], num_waypoints, replace=False)
        coords = np.column_stack((coords // MAX_X, coords % MAX_Y))
        return Grid(coords, charge_station)
    
    def distance_between(self, first_position: np.array, second_position: np.array):
        return np.abs(first_position - second_position).sum()

    def distance_from_charge_station(self, position: np.array):
        return self.distance_between(position, self.charge_station)

    def __distances_from_waypoints(self, position: np.array):
        return np.abs(self.waypoints - position).sum(axis=1)

    def nearest_waypoint(self, position: np.array):
        return self.waypoints[self.__distances_from_waypoints(position).argmin()]
    
    def direction_to(self, start: np.array, arrival: np.array):
        dx, dy = arrival - start

        if abs(dx) > abs(dy):
            return np.array([np.sign(dx), 0])
        return np.array([0, np.sign(dy)])

    def direction_to_charge_station(self, position: np.array):
        return self.direction_to(position, self.charge_station)
    
    def direction_to_nearest_waypoint(self, position: np.array):
        return self.direction_to(position, self.nearest_waypoint(position))
    
    def __waypoint_index(self, position: np.array):
        return np.where(np.all(self.waypoints == position, axis=1))
    
    def __over_charge_station(self, position: np.array):
        return np.all(self.charge_station == position)

    def all_waypoints_visited(self):
        return self.waypoints_counter == 0

    def compute_reward(self, robot: Robot):
        if self.__over_charge_station(robot.position):
            if self.all_waypoints_visited():
                self.episode_continues = False
                self.mission_complete = True
                return COMPLETE_REWARD + CHARGE_REWARD
            else:
                robot.recharge()
                return CHARGE_REWARD

        if not robot.can_move():
            self.episode_continues = False
            return FAIL_REWARD
        
        i = self.__waypoint_index(robot.position)[0]
        is_hovering_waypoint = len(i) > 0

        if is_hovering_waypoint:
            i = i.item()
            self.waypoints = np.delete(self.waypoints, i, axis=0)
            self.waypoints_counter -= 1
            self.waypoints_status += 2 ** i
            return WAYPOINT_REWARD

        return EMPTY_REWARD

    def draw_charge_station(self, screen):
        x, y = self.charge_station
        pg.draw.rect(screen, GREEN, (x*SIZE+1, y*SIZE+1, SIZE-1, SIZE-1))
    
    def draw_waypoints(self, screen):
        for w in self.waypoints:
            x, y = w
            pg.draw.rect(screen, BLUE, (x*SIZE+1, y*SIZE+1, SIZE-1, SIZE-1))
    
    def has_waypoint_at(self, x, y):
        if self.all_waypoints_visited():
            return False
        ds = self.__distances_from_waypoints(np.array([x,y]))
        return (ds == 0).any()