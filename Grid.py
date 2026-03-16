import numpy as np
from Robot import Robot
from config import *
import pygame as pg

class Grid:
    def __init__(self, waypoints: np.array, charge_station = np.zeros(2, 'int32')):
        self.reward_matrix = np.ones((MAX_Y, MAX_X), dtype='int32') * EMPTY_REWARD
        self.reward_matrix[tuple(charge_station)] = CHARGE_REWARD
        self.waypoints_status = 0

        self.waypoints = waypoints
        self.charge_station = charge_station
        self.episode_continues = True
        self.waypoints_counter = waypoints.shape[0]

        for w in waypoints:
            self.reward_matrix[tuple(w)] = WAYPOINT_REWARD

    
    def random_generate(N, charge_station = np.zeros(2, 'int32')):
        coords = np.random.choice(np.arange(1, MAX_X * MAX_Y), N, replace=False)
        coords = np.column_stack((coords // MAX_X, coords % MAX_Y))
        return Grid(coords, charge_station)
    
    def distances_from_new_waypoints(self, position: np.array):
        return np.abs(self.waypoints - position).sum(axis=1)
    
    def distance_between(self, first_position: np.array, second_position: np.array):
        return np.abs(first_position - second_position).sum()

    def distance_from_charge_station(self, position: np.array):
        return self.distance_between(position, self.charge_station)

    def nearest_new_waypoint(self, position: np.array):
        return self.waypoints[self.distances_from_new_waypoints(position).argmin()]
    
    def waypoint_index(self, position: np.array):
        return np.where(np.all(self.waypoints == position, axis=1))
    
    def is_charge_station(self, position: np.array):
        return np.all(self.charge_station == position)

    def all_waypoints_visited(self):
        return self.waypoints_counter == 0

    def border_penalty(self, position: np.array):
        x, y = position
        if x == 0:
            if y == 0:
                return 0
            return -10
        
        if x == WIDTH - 1 or y == 0 or y == HEIGTH - 1:
            return -10
        return 0

    def too_far_from_charge_station_reward(self, robot: Robot):
        if self.distance_from_charge_station(robot.position) > robot.battery:
            return -20
        return 0

    def compute_reward(self, robot: Robot):
        reward = self.too_far_from_charge_station_reward(robot) + self.border_penalty(robot.position)

        if self.is_charge_station(robot.position):
            if self.all_waypoints_visited():
                self.episode_continues = False
                print('Mission accomplished')
                return CHARGE_REWARD + COMPLETE_REWARD + reward
            else:
                robot.recharge()
                return CHARGE_REWARD + reward

        if not robot.can_move():
            self.episode_continues = False
            return FAIL_REWARD + reward
        
        i = self.waypoint_index(robot.position)[0]
        is_hovering_waypoint = len(i) > 0

        if is_hovering_waypoint:
            i = i.item()
            self.waypoints = np.delete(self.waypoints, i, axis=0)
            self.waypoints_counter -= 1
            self.waypoints_status += 2 ** i
            return WAYPOINT_REWARD + reward
        
        if self.all_waypoints_visited():
            target_distance = self.distance_from_charge_station(robot.position)
        else:
            target_distance = self.distance_between(robot.position, self.nearest_new_waypoint(robot.position))

        return reward + WAYPOINT_REWARD/((target_distance + 1)**2)

    def draw_charge_station(self, screen):
        x, y = self.charge_station
        pg.draw.rect(screen, GREEN, (x*SIZE+1, y*SIZE+1, SIZE-1, SIZE-1))
    
    def draw_waypoints(self, screen):
        for w in self.waypoints:
            x, y = w
            pg.draw.rect(screen, BLUE, (x*SIZE+1, y*SIZE+1, SIZE-1, SIZE-1))
    
    """
    def direction_to(self, arrival: np.array, start: np.array):
        dx, dy = arrival - start

        if abs(dx) > abs(dy):
            return np.array([np.sign(dx), 0])
        else:
            return np.array([0, np.sign(dy)])

    def direction_to_nearest_new_waypoint(self, position: np.array):
        return self.direction_to(arrival=self.nearest_new_waypoint(position), start=position)
    
    def direction_to_charge_station(self, position: np.array):
        return self.direction_to(arrival=self.charge_station, start=position)
"""