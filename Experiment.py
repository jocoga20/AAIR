from ValueFunction import ValueFunction
from utils import *
import random
import numpy as np
from render import *

class Experiment:
    def __init__(self, num_waypoints: int, value_function: ValueFunction, charge_station = np.zeros(2, 'int32')):
        self.num_waypoints = num_waypoints
        self.value_function = value_function
        self.charge_station = charge_station
    
    def __set_seed(self, seed):
        """
        Sets all the seeds needed to reproduce the experiment.
        """
        random.seed(seed)
        np.random.seed(seed)
    
    def __init_ambient(self) -> tuple[Grid, Robot]:
        """
        This function returns a Grid and a Robot instance.
        Modifiy this function to customize the beginning of the experiment.
        """
        x, y = self.charge_station
        grid = Grid.random_generate(self.num_waypoints, self.charge_station)
        robot = Robot(x=x, y=y, full_battery=FULL_BATTERY)
        return grid, robot
    
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

    def run(self, seed: int, policy, render: NoRender, pmax: float = 0.9):
        self.__set_seed(seed)
        grid, robot = self.__init_ambient()
        s0 = state_key(robot, grid)
        score = 0
        
        while grid.episode_continues and not render.check_events().quitted_pygame():
            if render.paused():
                continue
            render.before_move(grid, robot)
            
            direction = policy(grid, robot)
            direction = self.choose_direction(direction, robot.position, pmax=pmax)
            robot.move(direction)

            reward = grid.compute_reward(robot)
            score += self.value_function.reward_discount * reward
            s1 = state_key(robot, grid)
            self.value_function.update(s0, s1, reward)
            s0 = s1
            render.after_move(grid, robot)
        
        return score, grid.mission_complete