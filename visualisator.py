import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt


class Visualisator():

    def __init__(self, r):
        self.r = r

    def __normL(self, y, yy, norm=2):
        delta = y - yy
        return np.linalg.norm(delta, norm)

    def histo(self, norm=2):
        x = []
        for i in range(self.r.c):
            ry = self.r.real_y[i]
            py = self.r.predicted_y[i]
            n = self.__normL(ry, py, norm)
            x.append(n)
        plt.hist(x, bins=30)
        plt.show()
