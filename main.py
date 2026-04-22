import numpy as np
from ValueFunction import ValueFunction
from ValueFunctionLambda import ValueFunctionLambda
from config import *
from utils import *
from experiment import *


import matplotlib.pyplot as plt

def plot_hist(values, bins=20):
    values = np.array(values)
    plt.figure()
    plt.scatter(list(range(len(values))), values, s=1)
    plt.axhline(values.mean(), linestyle="dashed", label=f"mean = {values.mean():.3f}")
    
    plt.legend()
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.title("Histogram")
    plt.show()
#    plt.savefig(f'hist{bins}.png')

vf = ValueFunctionLambda(step_size_lambda=STEP_SIZE_RULE, reward_discount=REWARD_DISCOUNT)
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

for it in range(100):
    experiment(seed=42 + it, value_function=vf, num_waypoints=5)

xs = list(vf.value_dict.values())
xs = sorted(xs)

plot_hist(xs, bins=40)