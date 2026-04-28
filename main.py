import numpy as np
from ValueFunction import ValueFunction
from ValueFunctionLambda import ValueFunctionLambda
from config import *
from utils import *
from Experiment import *
import policies


import matplotlib.pyplot as plt

def plot_values(values, bins=20):
    values = np.array(values)
    plt.figure()
#    plt.hist(values, bins)
    plt.scatter(list(range(len(values))), values, s=1)
    plt.axhline(values.mean(), linestyle="dashed", label=f"mean = {values.mean():.3f}", color='red')
    
    plt.legend()
    plt.show()
#    plt.savefig(f'hist{bins}.png')

import numpy as np
from sklearn.decomposition import PCA

def plot_vf(vf: ValueFunction):
    xs = np.array(list(vf.value_dict.keys()))
    print(xs.shape)
    pca = PCA(3)
    xs = pca.fit_transform(xs)
    print(f'Explained variance: {pca.explained_variance_ratio_}')

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    v = vf.value_dict.values()
    vma = max(v)
    vmi = min(v)
    sc = ax.scatter(xs[:, 0], xs[:, 1], xs[:, 2], c=v, cmap='inferno', vmin=vmi, vmax=vma)
    print(vmi, vma)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    fig.colorbar(sc, ax=ax)
    plt.show()


vf = ValueFunction(step_size_lambda=STEP_SIZE_RULE, reward_discount=REWARD_DISCOUNT)
vf.init_state_monitor([(0,0,FULL_BATTERY,0), (1,1,16,17), (4,19,57,24), (16,6,38,16)])
ex = Experiment(num_waypoints=5, value_function=vf)
render = DrawRender()
for it in range(500):
    render.set_title(f'Seed {42+it}')
    ex.run(42 + it, policies.pedant_policy, render)

print(vf.monitored_states.values())