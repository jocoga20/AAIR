from tqdm import tqdm

from ValueFunction import ValueFunction
from ValueFunctionLambda import ValueFunctionLambda
from config import *
from utils import *
from Experiment import *
import policies

vf = ValueFunction(step_size_lambda=step_size_rule, reward_discount=REWARD_DISCOUNT)
#most_visiteds = [(0, 0, 80, 0), (0, 1, 79, 0), (1, 0, 79, 0), (0, 2, 78, 0)]
#vf.init_state_monitor(most_visiteds)

ex = Experiment(num_waypoints=5, value_function=vf)

norender = NoRender()

print('Pre learn')
for it in tqdm(range(10_000)):
    ex.run(seed=42, policy=policies.pedant_policy, render=norender)

import pickle
def save(name, obj):
    with open(name, 'wb') as f:
        pickle.dump(obj, f)

def load(name):
    with open(name, 'rb') as f:
        return pickle.load(f)

save('vf.pkl', vf)
vf = load('vf.pkl')
exit()
tprender = TimePlotRender(vf)
print('Showing')
for it in range(10):
    ex.run(seed=42, policy=policies.pedant_policy, render=tprender)