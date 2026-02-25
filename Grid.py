import numpy as np
from Robot import Robot
from config import *

class Grid:
    def __init__(self, waypoints: np.array, charge_station = np.zeros(2, 'int32'), width = 20, heigth = 20):
        self.reward_matrix = np.ones((heigth, width), dtype='int32') * EMPTY_REWARD
        self.reward_matrix[tuple(charge_station)] = CHARGE_REWARD

        for w in waypoints:
            self.reward_matrix[tuple(w)] = WAYPOINT_REWARD

        self.waypoints = waypoints
        self.not_visited = np.ones(waypoints.shape[0], dtype='bool')
        self.charge_station = charge_station
        self.episode_continues = True
    
    def random_generate(N, charge_station = np.zeros(2, 'int32'), width = 20, heigth = 20):
        return Grid(np.random.randint(20, size=(N, 2)), charge_station, width, heigth)
    
    def nearest_waypoint(self, position: np.array):
        return np.abs(self.waypoints[self.not_visited] - position).sum(axis=1).argmin()

    def direction_to_nearest_waypoint(self, position: np.array):
        dx, dy = self.nearest_waypoint(position) - position

        if abs(dx) > abs(dy):
            return np.array([np.sign(dx), 0])
        else:
            return np.array([0, np.sign(dy)])
    
    def waypoint_index(self, position: np.array):
        return np.where(np.all(self.waypoints[self.not_visited] == position, axis=1))
    
    def is_charge_station(self, position: np.array):
        return np.all(self.charge_station, position)
    
    def all_waypoints_visited(self):
        return np.any(self.not_visited)

    def compute_reward(self, robot: Robot):
        if self.is_charge_station(robot.position):
            robot.recharge()
            return CHARGE_REWARD
        
        i = self.waypoint_index(robot.position)[0]

        if i.size != 0: # is over a waypoint of index i
            self.not_visited[i.astype('int32')] = False

            if self.all_waypoints_visited():
                self.episode_continues = False
                return WAYPOINT_REWARD + COMPLETE_REWARD
            else:
                if robot.can_move():
                    return WAYPOINT_REWARD
                else:
                    self.episode_continues = False
                    return FAIL_REWARD
        return EMPTY_REWARD
    
np.random.seed(42)
g = Grid.random_generate(N=5)
print(g.visit(np.array([5, 19])))