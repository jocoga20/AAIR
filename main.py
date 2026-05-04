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
most_visiteds = [(0, 0, 80, 9), (0, 0, 80, 0), (7, 6, 13, 31), (11, 1, 68, 0)]
vf.init_state_monitor(most_visiteds)

ex = Experiment(num_waypoints=5, value_function=vf)
render = DrawRenderValueFunction(vf)


for it in range(10):
#    render.set_title(f'Seed {42+it}')
    ex.run(42 + it, policies.pedant_policy, render)

def key_to_title(key):
    x, y, b, w = key
    w = "{0:b}".format(w)
    w = w[::-1]
    return (x, y, b, w)

def plot_state_values(i, key, values):
    plt.plot(values)
    plt.title(f'{i} {key_to_title(key)}')
    plt.show()

for i, (k, v) in enumerate(vf.monitored_states.items()):
    plot_state_values(i, k, v)