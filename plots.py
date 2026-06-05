import os
from matplotlib import pyplot as plt

def key_to_title(key):
    x, y, b, w = key
    return (x, y, b, w)

def plot_state_values(key, values, path=None):
    """
    If path is None it shows the plot, otherwise it saves it in path.
    """
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