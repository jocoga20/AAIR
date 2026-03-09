import numpy as np
from Robot import Robot
from config import *

class Grid:
    def __init__(self, waypoints: np.array, charge_station = np.zeros(2, 'int32'), width = 20, heigth = 20):
        self.reward_matrix = np.ones((heigth, width), dtype='int32') * EMPTY_REWARD
        self.reward_matrix[tuple(charge_station)] = CHARGE_REWARD
        self.waypoints_status = 0

        for w in waypoints:
            self.reward_matrix[tuple(w)] = WAYPOINT_REWARD

        self.waypoints = waypoints
        self.charge_station = charge_station
        self.episode_continues = True
    
    def random_generate(N, charge_station = np.zeros(2, 'int32'), width = 20, heigth = 20):
        coords = np.random.choice(np.arange(1, width * heigth), N, replace=False)
        coords = np.column_stack((coords // 20, coords % 20))
        return Grid(coords, charge_station, width, heigth)
    
    def distances_from_new_waypoints(self, position: np.array):
        return np.abs(self.waypoints - position).sum(axis=1)
    
    def distance_between(self, first_position: np.array, second_position: np.array):
        return np.abs(first_position - second_position).sum()

    def distance_from_charge_station(self, position: np.array):
        return self.distance_between(position, self.charge_station)

    def nearest_new_waypoint(self, position: np.array):
        return self.waypoints[self.distances_from_new_waypoints(position).argmin()]
    
    def distance_to_furthest_waypoint(self, position: np.array):
        return self.distances_from_new_waypoints(position).max()

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
    
    def waypoint_index(self, position: np.array):
        return np.where(np.all(self.waypoints == position, axis=1))
    
    def is_charge_station(self, position: np.array):
        return np.all(self.charge_station == position)
    
    def all_waypoints_visited(self):
        return len(self.waypoints) == 0

    def compute_reward(self, robot: Robot):
        if self.is_charge_station(robot.position):
            if self.all_waypoints_visited():
                self.episode_continues = False
                return CHARGE_REWARD + COMPLETE_REWARD
            else:
                robot.recharge()
                return CHARGE_REWARD

        if not robot.can_move():
            self.episode_continues = False
            return FAIL_REWARD
        
        i = self.waypoint_index(robot.position)[0]
        is_hovering_waypoint = len(i) > 0

        if is_hovering_waypoint:
            i = i.item()
            self.waypoints = np.delete(self.waypoints, i, axis=0)
            self.waypoints_status += 2 ** i
            return WAYPOINT_REWARD
        
        return EMPTY_REWARD