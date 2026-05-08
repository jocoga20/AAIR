from tqdm import tqdm

from ValueFunction import ValueFunction
from ValueFunctionLambda import ValueFunctionLambda
from config import *
from utils import *
from Experiment import *
import policies

vf = ValueFunction(step_size_lambda=STEP_SIZE_RULE, reward_discount=REWARD_DISCOUNT)
most_visiteds = [(0, 0, 80, 0), (0, 1, 79, 0), (1, 0, 79, 0), (0, 2, 78, 0)]
vf.init_state_monitor(most_visiteds)

ex = Experiment(num_waypoints=5, value_function=vf)

norender = NoRender()
print('Pre learn')
for it in tqdm(range(10_000)):
    ex.run(42, policies.pedant_policy, norender)

# render = DrawRender()
# print('Showing')
# for it in range(10):
#     ex.run(42, policies.pedant_policy, render)

print('Plot')
for i, (k, v) in enumerate(vf.monitored_states.items()):
    plot_state_values(i, k, v)
