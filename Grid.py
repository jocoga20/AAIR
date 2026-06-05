import numpy as np
from Robot import Robot
from config import *
import pygame as pg

class Grid:
    def __init__(self, waypoints: np.ndarray, charge_station = np.zeros(2, 'int32')):
        self.waypoints = waypoints
        self.waypoints_mask = np.full(shape=waypoints.shape[0], fill_value=True, dtype=bool)
        self.unseen_waypoints_count = waypoints.shape[0]

        self.charge_station = charge_station
        self.episode_continues = True
        self.mission_complete = False

    @staticmethod
    def __plane_pos_to_index(pos: np.array):
        x, y = pos
        return y * MAX_Y + x

    @staticmethod
    def random_generate(num_waypoints: int, seed: int, charge_station = np.zeros(2, 'int32')):
        np.random.seed(seed)
        charge_station_idx = Grid.__plane_pos_to_index(charge_station)
        indexes = np.arange(0, MAX_X * MAX_Y)
        indexes = np.random.choice(indexes[indexes != charge_station_idx], num_waypoints, replace=False)
        indexes = np.column_stack((indexes // MAX_X, indexes % MAX_Y))
        return Grid(indexes, charge_station)
    
    def distance_between(self, first_position: np.array, second_position: np.array):
        return np.abs(first_position - second_position).sum()

    def distance_from_charge_station(self, position: np.array):
        return self.distance_between(position, self.charge_station)

    def __distances_from_waypoints(self, position: np.array):
        return np.abs(self.waypoints[self.waypoints_mask] - position).sum(axis=1)

    def nearest_waypoint(self, position: np.array):
        return (self.waypoints[self.waypoints_mask])[self.__distances_from_waypoints(position).argmin()]
    
    def direction_to(self, start: np.array, arrival: np.array):
        dx, dy = arrival - start

        if abs(dx) > abs(dy):
            return np.array([np.sign(dx), 0])
        return np.array([0, np.sign(dy)])

    def direction_to_charge_station(self, position: np.array):
        return self.direction_to(position, self.charge_station)
    
    def direction_to_nearest_waypoint(self, position: np.array):
        return self.direction_to(position, self.nearest_waypoint(position))
    
    def __over_charge_station(self, position: np.array):
        return np.all(self.charge_station == position)

    def all_waypoints_visited(self):
        return self.unseen_waypoints_count == 0

    def __waypoint_index_at(self, position: np.array):
        matches = np.all(self.waypoints[self.waypoints_mask] == position, axis=1)
        return np.argmax(matches) if np.any(matches) else -1

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
        
        if self.all_waypoints_visited():
            return EMPTY_REWARD
        
        mask_wi = self.__waypoint_index_at(robot.position)
        if mask_wi < 0:
            return EMPTY_REWARD

        idxs = np.where(self.waypoints_mask)[0]
        real_idx = idxs[mask_wi]
        self.waypoints_mask[real_idx] = False
        self.unseen_waypoints_count -= 1

        return WAYPOINT_REWARD

    def draw_charge_station(self, screen):
        x, y = self.charge_station
        pg.draw.rect(screen, GREEN, (x*SIZE+1, y*SIZE+1, SIZE-1, SIZE-1))
    
    def draw_waypoints(self, screen):
        for w in self.waypoints[self.waypoints_mask]:
            x, y = w
            pg.draw.rect(screen, BLUE, (x*SIZE+1, y*SIZE+1, SIZE-1, SIZE-1))
    
    def has_waypoint_at(self, x, y):
        if self.all_waypoints_visited(): return False
        return self.__waypoint_index_at(position=np.array([x,y])) >= 0
    
    def waypoints_status(self):
        # swapped 0 and 1 for view purposes
        return ''.join(np.where(self.waypoints_mask, '0', '1'))

    def reset(self):
        self.waypoints_mask.fill(True)
        self.unseen_waypoints_count = self.waypoints.shape[0]
        self.episode_continues = True
        self.mission_complete = False