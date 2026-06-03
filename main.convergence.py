from tqdm import tqdm

from Experiment import Experiment
from ValueFunction import ValueFunction
from ValueFunctionLambda import ValueFunctionLambda
from plots import plot_state_values
import policies
import render

vfl = ValueFunctionLambda(reward_discount=0.9)

no_render = render.NoRender()
ex = Experiment(num_waypoints=5, value_function=vfl)

for _ in tqdm(range(500)):
    ex.run(seed=42, policy=policies.greedy_policy, render=no_render)

for k, v in vfl.values_dict.items():
    if k[-1] == 0:
        pre_path = '0wps_temp'
    else:
        pre_path = 'n0wps_temp'
    plot_state_values(key=k, values=v, path=f'imgs/{pre_path}/{k}.svg')