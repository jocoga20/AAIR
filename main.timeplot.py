from tqdm import tqdm

from Experiment import Experiment
from ValueFunction import ValueFunction
from ValueFunctionLambda import ValueFunctionLambda
from policies import *
from render import *

vf = ValueFunction(reward_discount=0.9)
no_render = NoRender()
ex = Experiment(grid_seed=42, num_waypoints=5, value_function=vf)

for it in tqdm(range(100)):
    ex.run(robot_seed=42+it, policy=greedy_policy, render=no_render, pmax=0.8)

ex.run(robot_seed=42, policy=greedy_policy, render=TimePlotRender(value_function=vf, frame_draw_time=.4), pmax=0.8)