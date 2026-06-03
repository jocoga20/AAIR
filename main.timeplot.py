from tqdm import tqdm

from Experiment import Experiment
from ValueFunction import ValueFunction
from ValueFunctionLambda import ValueFunctionLambda
from fileutils import *
import policies
from render import *
vf = ValueFunction(reward_discount=0.9)#.load('vfs/vf.09rd.10k.gp.pkl')

no_render = NoRender()
ex = Experiment(num_waypoints=5, value_function=vf)
for it in tqdm(range(5_000)):
    ex.run(seed=42+it, policy=policies.greedy_policy, render=no_render)
