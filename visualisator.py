import numpy as np
from result import Result
import tools
import matplotlib
import math
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt


class Visualisator():

    def __init__(self):
        pass

    def __normL(self, y, yy, norm=2):
        delta = y - yy
        return np.linalg.norm(delta, norm)

    def histo(self, r, bins=30):
        n_1 = []
        n_inf = []
        for i in range(self.r.c):
            ry = r.real_y[i]
            py = r.predicted_y[i]
            n_1.append(self.__normL(ry, py, 1) / r.n)
            n_inf.append(self.__normL(ry, py, math.inf))
        plt.hist(n_1, bins, alpha=0.5, label='Norm 1')
        plt.hist(n_inf, bins, alpha=0.5, label='Norm inf')
        plt.legend(loc='upper right')
        plt.title("Test")
        plt.show()

    def compare(self, predictors, env_names, agent_names, cs, norm=1, bins=30):
        for predictor_name in predictors:
            for env_name in env_names:
                for agent_name in agent_names:
                    for c in cs:
                        filename = tools.FileNaming.resultName(
                            predictor_name, env_name, agent_name, c
                        )
                        r = Result(filename=filename)
                        n = []
                        for i in range(r.c):
                            ry = r.real_y[i]
                            py = r.predicted_y[i]
                            n.append(self.__normL(ry, py, 1) / r.n)
                        label = predictor_name
                        label += " - " + env_name + " - "
                        label += agent_name + " - " + str(c)
                        plt.hist(n, bins, alpha=0.5, label=label)
        plt.legend(loc='upper right')
        plt.title("Test")
        plt.show()
