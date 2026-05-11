from Grid import Grid
from Robot import Robot
from config import *
from ValueFunction import ValueFunction
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def state_key(robot: Robot, grid: Grid):
    """
    This function sums up the state of the ambient (agent included) in a key to access and manipulate the value function
    """
    x, y = robot.position
    x, y = x.item(), y.item()

    return (x, y, robot.battery, grid.waypoints_status)

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

def key_to_title(key):
    x, y, b, w = key
    w = "{0:b}".format(w)
    w = w[::-1]
    return (x, y, b, w)

def plot_state_values(i, key, values, path=None):
    plt.plot(values)
    plt.title(f'{i} {key_to_title(key)}')
    if path is None:
        plt.show()
    else:
        print(f'Saved {path}')
        plt.savefig(path)
    plt.clf()