import numpy as np
import matplotlib
import math
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt


class Visualisator():

    def __init__(self, r):
        self.r = r

    def __normL(self, y, yy, norm=2):
        delta = y - yy
        return np.linalg.norm(delta, norm)

    def histo(self, bins=30):
        n_1 = []
        n_inf = []
        for i in range(self.r.c):
            ry = self.r.real_y[i]
            py = self.r.predicted_y[i]
            n_1.append(self.__normL(ry, py, 1) / self.r.n)
            n_inf.append(self.__normL(ry, py, math.inf))
        plt.hist(n_1, bins, alpha=0.5, label='Norm 1')
        plt.hist(n_inf, bins, alpha=0.5, label='Norm inf')
        plt.legend(loc='upper right')
        plt.title("Test")
        plt.show()
