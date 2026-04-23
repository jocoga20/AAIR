from ValueFunction import ValueFunction
from policies import pedant_policy as mypolicy
from utils import *
from time import sleep

class Experiment:
    def __init__(self, num_waypoints: int, value_function: ValueFunction, charge_station = np.zeros(2, 'int32')):
        self.num_waypoints = num_waypoints
        self.value_function = value_function
        self.charge_station = charge_station
    
    def set_seed(self, seed):
        """
        Sets all the seeds needed to reproduce the experiment.
        """
        random.seed(seed)
        np.random.seed(seed)
    
    def init_ambient(self, num_waypoints: int, charge_station: np.array):
        """
        This function returns a Grid and a Robot instance.
        Modifiy this function to customize the beginning of the experiment.
        """
        x, y = charge_station
        robot = Robot(x=x, y=y, full_battery=(MAX_X + MAX_Y)*2)
        grid = Grid.random_generate(num_waypoints, charge_station)
        return grid, robot
    
    def is_allowed(p):
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

    def run(self, seed: int, policy):
        self.set_seed(seed)
        grid, robot = self.init_ambient(self.num_waypoints, self.charge_station)
        s0 = state_key(robot, grid)
        score = 0

        while grid.episode_continues:
            direction = policy(grid, robot)
            direction = self.choose_direction(policy)
            robot.move(direction)

            reward = grid.compute_reward(robot)
            score += REWARD_DISCOUNT * reward
            s1 = state_key(robot, grid)
            self.value_function.update(s0, s1, reward)
            s0 = s1
        return score, grid.mission_complete

    def init_graphics(self, title):
        """
        This function only runs when in drawing mode. 
        """
        pg.init()
        screen = pg.display.set_mode((WIDTH, HEIGTH))
        pg.display.set_caption(title)
        screen.fill(WHITE)

        for y in range(0, HEIGTH, SIZE):
            pg.draw.line(screen, BLACK, (0, y), (WIDTH, y))
            
        for x in range(0, WIDTH, SIZE):
            pg.draw.line(screen, BLACK, (x, 0), (x, HEIGTH))
        return screen

    def quitted_pygame(self):
        for event in pg.event.get():
            if event.type == pg.QUIT:
                pg.quit()
                return True
        return False

    def run_draw(self, seed: int, policy, title = None):
        if title is None:
            title = f'Seed {seed}'
        grid, robot = self.init_ambient(self.num_waypoints, self.charge_station)
        s0 = state_key(robot, grid)
        score = 0

        screen = self.init_graphics()

        while grid.episode_continues and not self.quitted_pygame():
            robot.erase(screen)
            grid.draw_charge_station(screen)
            grid.draw_waypoints(screen)

            direction = mypolicy(grid, robot)
            direction = self.choose_direction(direction, robot.position)
            robot.move(direction)

            robot.draw(screen)

            reward = grid.compute_reward(robot)
            score += REWARD_DISCOUNT * reward
            s1 = state_key(robot, grid)
            self.value_function.update(s0, s1, reward)
            s0 = s1

            pg.display.flip()
            sleep(FRAME_DRAW_TIMER)
        
        return score, grid.mission_complete
    
    def run_draw_number_grid(self, seed: int, policy, title = None):
        if title is None:
            title = f'Seed {seed}'
            # TODO:
            # - refactor intelligente per experiment e experiment draw
            # - griglia value function
            # - k stati di riferimento e plotto come evolve nel corso degli episodi (per mostrare convergenza)