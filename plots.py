import os

from matplotlib import pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

from ValueFunction import ValueFunction


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

def plot_state_values(key, values, path=None):
    plt.plot(values)
    plt.title(key_to_title(key))
    plt.xlabel('episode')
    plt.ylabel('value')
    if path is None:
        plt.show()
    else:
        dirname = os.path.dirname(path)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        print(f'Saved {path}')
        plt.savefig(path)
    plt.clf()