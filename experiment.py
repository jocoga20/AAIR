from ValueFunction import ValueFunction
from policies import secure_policy as mypolicy
from utils import *
from time import sleep

def experiment(num_waypoints: int, seed: int, value_function: ValueFunction, charge_station_position = np.zeros(2, 'int32')):
    set_seed(seed)
    grid, robot = init_ambient(num_waypoints, charge_station_position)
    s0 = state_key(robot, grid)
    score = 0

    while grid.episode_continues:
        direction = mypolicy(grid, robot)
        direction = choose_direction(direction, robot.position)
        robot.move(direction)

        reward = grid.compute_reward(robot)
        score += REWARD_DISCOUNT * reward
        s1 = state_key(robot, grid)
        value_function.update(s0, s1, reward)
        s0 = s1

    return score, grid.mission_complete

def experiment_draw(num_waypoints: int, seed: int, value_function: ValueFunction, charge_station_position = np.zeros(2, 'int32'), title = None):
    if title is None:
        title = f'Seed {seed}'
    set_seed(seed)
    grid, robot = init_ambient(num_waypoints, charge_station_position)
    s0 = state_key(robot, grid)
    score = 0

    screen = init_graphics(title)

    while grid.episode_continues and not quitted_pygame():
        robot.erase(screen)
        grid.draw_charge_station(screen)
        grid.draw_waypoints(screen)

        direction = mypolicy(grid, robot)
        direction = choose_direction(direction, robot.position)
        robot.move(direction)

        robot.draw(screen)

        reward = grid.compute_reward(robot)
        score += REWARD_DISCOUNT * reward
        s1 = state_key(robot, grid)
        value_function.update(s0, s1, reward)
        s0 = s1

        pg.display.flip()
        sleep(FRAME_DRAW_TIMER)

    return score, grid.mission_complete