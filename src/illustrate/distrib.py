import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.stats import chi2
from scipy.stats import norm
from math import sqrt


def genPoint():
    x = np.random.normal()
    y = np.random.normal()
    return x, y


def plot(n=50000, p=0.95):

    fig, ax = plt.subplots()

    points = [genPoint() for _ in range(n)]
    X = [point[0] for point in points]
    Y = [point[1] for point in points]
    plt.scatter(X, Y, c='b', s=1)

    alpha = norm.ppf(0.5 * (1. + p))
    r = sqrt(chi2.ppf(p**2, df=2))
    l = 3

    circle = plt.Circle((0, 0), r, color='black',
                        fill=False, linewidth=2)

    rect = patches.Rectangle((-alpha, -alpha), 2 * alpha,
                             2 * alpha, linewidth=2,
                             edgecolor='black', facecolor='none')

    ax.add_patch(rect)

    ax.set_xlim((-l, l))
    ax.set_ylim((-l, l))
    ax.add_artist(circle)

    plt.show()
