from Experiment import Experiment
from ValueFunction import ValueFunction
from ValueFunctionLambda import ValueFunctionLambda
from fileutils import *
import policies
import render
vf = ValueFunction(reward_discount=0.9).load('vfs/vf.09rd.10k.gp.pkl')

ex = Experiment(num_waypoints=5, value_function=vf)
ex.run(seed=42, policy=policies.greedy_policy, render=render.TimePlotRender(vf, frame_draw_time=0.8))