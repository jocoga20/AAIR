from Experiment import Experiment
from fileutils import *
import policies
import render
vfl = load('vfs/vf0.45rd09.10k.gp.pkl')

ex = Experiment(num_waypoints=5, value_function=vfl)
ex.run(seed=42, policy=policies.greedy_policy, render=render.TimePlotRender(vfl, frame_draw_time=1.5))