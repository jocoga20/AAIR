from tqdm import tqdm

from Experiment import Experiment
from ValueFunction import ValueFunction
from ValueFunctionLambda import ValueFunctionLambda
from policies import *
from render import *

vf = ValueFunction(reward_discount=0.9)#.load()
no_render = NoRender()
ex = Experiment(num_waypoints=5, value_function=vf)

for it in tqdm(range(100)):
    ex.run(seed=42+it, policy=greedy_policy, render=no_render, pmax=0.8)

#ex.run(seed=42, policy=greedy_policy, render=TimePlotRender(vf, .5), pmax=.8)

print(len(vf.values_dict))