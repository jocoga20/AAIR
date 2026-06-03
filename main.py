from tqdm import tqdm

from ValueFunction import ValueFunction
from ValueFunctionLambda import ValueFunctionLambda
from config import *
from utils import *
from Experiment import *
import policies
from fileutils import *

vf = ValueFunction(reward_discount=0.9)
most_visiteds = [(0, 0, 80, 0), (0, 1, 79, 0), (1, 0, 79, 0), (0, 2, 78, 0)]
vf.init_state_monitor(most_visiteds)

ex = Experiment(num_waypoints=5, value_function=vf)

norender = NoRender()
vfs = [lambda: ValueFunction(reward_discount=0.9), lambda: ValueFunctionLambda(reward_discount=0.9)]
pols = [('gp', policies.greedy_policy), ('pp', policies.pedant_policy), ('sp', policies.secure_policy)]
for i, vfbuild in enumerate(vfs):
    for pol_name, pol in pols:
        vf = vfbuild()
        ex = Experiment(num_waypoints=5, value_function=vf)
        for it in tqdm(range(1)):
            ex.run(seed=42, policy=pol, render=norender)
        if i == 1:
            lam = vf.eligibility.decay
            print(lam)
        else:
            lam = ''
        #save(f'vfs/vf{lam}rd09.10k.{pol_name}.pkl', vf)
exit()

# print('Pre learn')
# for it in tqdm(range(10_000)):
#     ex.run(seed=42, policy=policies.pedant_policy, render=norender)
from fileutils import *

vf = load('vf.pkl')

tprender = TimePlotRender(vf)
print('Showing')
for it in range(10):
    ex.run(seed=42, policy=policies.pedant_policy, render=tprender)