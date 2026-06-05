from ValueFunction import ValueFunction
from utils import *
import random
import numpy as np
from render import *

class Experiment:
    def __init__(self, grid_seed: int, num_waypoints: int, value_function: ValueFunction, charge_station = np.zeros(2, 'int32')):
        self.num_waypoints = num_waypoints
        self.value_function = value_function
        self.charge_station = charge_station

        self.grid = Grid.random_generate(num_waypoints=num_waypoints, seed=grid_seed, charge_station=charge_station)
        x, y = charge_station
        self.robot = Robot(x, y, FULL_BATTERY)
    
    def is_allowed(self, p: np.array):
        x, y = p
        return 0 <= x and x < MAX_X and 0 <= y and y < MAX_Y

    def choose_direction(self, direction: np.array, robot_position: np.array, pmax: float = 0.9):
        """
        Chooses between the main direction and one or more orthogonal direction(s).
        A direction is allowed if it does not brings the robot outside the grid world.
        The main direction is always chosen with probability **pmax**.
        If there is one allowed orthogonal direction, probability of choosing it is **1 - pmax**.
        If there are two allowed orthogonal directions, probability of choosing each of them is **(1 - pmax)/2**.
        """
        dx, dy = direction
        d1 = np.array([dy, dx])
        d2 = -d1
        directions = [direction] + [d for d in [d1, d2] if self.is_allowed(d + robot_position)]
        probabilities = [pmax, 1-pmax] if len(directions) == 2 else [pmax, (1-pmax)/2, (1-pmax)/2]
        return random.choices(directions, probabilities)[0]

    def reset_ambient(self):
        self.grid.reset()
        self.robot.reset(self.charge_station)

    def run(self, robot_seed: int, policy, render: NoRender, pmax: float = 0.9):
        self.reset_ambient()
        random.seed(robot_seed)
        s0 = state_key(self.robot, self.grid)
        score = 0
        
        while self.grid.episode_continues and not render.check_events().quitted_pygame():
            if render.paused():
                continue

            render.before_move(self.grid, self.robot)
            
            direction = policy(self.grid, self.robot)
            direction = self.choose_direction(direction, self.robot.position, pmax=pmax)
            self.robot.move(direction)

            reward = self.grid.compute_reward(self.robot)
            score += self.value_function.reward_discount * reward
            s1 = state_key(self.robot, self.grid)
            self.value_function.update(s0, s1, reward)
            s0 = s1
            render.after_move(self.grid, self.robot)
        
        return score, self.grid.mission_complete