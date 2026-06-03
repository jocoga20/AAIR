import os

from matplotlib import pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

from ValueFunction import ValueFunction

def key_to_title(key):
    x, y, b, w = key
    w = "{0:b}".format(w)
    return (x, y, b, w)

def create_mid_dirs(path):
    dirname = os.path.dirname(path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)

def plot_state_values(key, values, path=None):
    plt.plot(values)
    plt.title(key_to_title(key))
    plt.xlabel('episode')
    plt.ylabel('value')

    if path is None:
        plt.show()
    else:
        create_mid_dirs(path)
        plt.savefig(path)
        print(f'Saved {path}')
        
    plt.clf()