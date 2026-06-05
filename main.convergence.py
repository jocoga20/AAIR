from tqdm import tqdm

from Experiment import Experiment
from ValueFunction import ValueFunction
from ValueFunctionLambda import ValueFunctionLambda
from plots import plot_state_values
from policies import *
from render import *

def save_state_history(values_dict: dict, path: str, min_vals: int):
    for k, v in values_dict.items():
        lv = len(v)
        if lv < min_vals:
            continue
        wps_count = k[-1].count('1')
        subpath = f'{wps_count}wps'
        
        plot_state_values(k, v, f'{path}/{subpath}/{k}.svg')

def train(vf: ValueFunction, policy, pmax=0.9, nepisodes=10_000):
    no_render = NoRender()
    ex = Experiment(grid_seed=42, num_waypoints=5, value_function=vf)
    for it in tqdm(range(nepisodes)):
        ex.run(robot_seed=42+it, policy=policy, pmax=pmax, render=no_render)

vf = ValueFunction(reward_discount=0.9)
train(vf=vf, policy=greedy_policy, nepisodes=1_000)
save_state_history(vf.values_dict, path='imgs', min_vals=50)