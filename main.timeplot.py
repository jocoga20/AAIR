import sys

from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm

from Experiment import Experiment
from ValueFunction import ValueFunction
from ValueFunctionLambda import ValueFunctionLambda
from fileutils import *
from policies import *
from render import *

vf = ValueFunction(reward_discount=0.9).load('vfs/tmp.pkl')
no_render = NoRender()
ex = Experiment(num_waypoints=5, value_function=vf)

for it in tqdm(range(10_000)):
    ex.run(seed=42, policy=greedy_policy, render=TimePlotRender(vf, 1.), pmax=0.8)

