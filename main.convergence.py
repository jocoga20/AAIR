from tqdm import tqdm

from Experiment import Experiment
from ValueFunctionLambda import ValueFunctionLambda
from plots import plot_state_values
import policies
import render


vfl = ValueFunctionLambda(reward_discount=0.9)
# states_to_monitor = (0, 0, 80, 0), (0, 1, 79, 0), (1, 0, 79, 0), (0, 2, 78, 0)
# vfl.init_state_monitor(states_to_monitor)

no_render = render.NoRender()
ex = Experiment(num_waypoints=5, value_function=vfl)

for _ in tqdm(range(10_000)):
    ex.run(seed=42, policy=policies.greedy_policy, render=no_render)

for k, v in vfl.monitored_states.items():
    if k[-1] == 0:
        pre_path = 'zero_wps'
    else:
        pre_path = 'nonzero_wps'
    plot_state_values(key=k, values=v, path=f'imgs/{pre_path}/{k}.svg')