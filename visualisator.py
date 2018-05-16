import numpy as np
from result import Result
import tools
import matplotlib.pyplot as plt
import math


class Visualisator():

    def __init__(self):
        pass

    def __normL(self, y, yy, norm=2):
        delta = y - yy
        return np.linalg.norm(delta, norm)

    def listNorm(self, r):
        n = []
        for i in range(len(r.real_y)):
            ry = r.real_y[i]
            py = r.predicted_y[i]
            n.append(self.__normL(ry, py, 1) / r.n)
            # print("Norm:", n[i], ", sigma:", r.sigma[i])
        return n

    def compare(self, predictors, env_names, agent_names, cs, norm=1, bins=100):
        nrows = 1
        ncols = 1
        n = len(env_names)
        if n > 1:
            ncols = 2
        if n > 8:
            ncols = 3
        nrows = math.ceil(n / ncols)

        plt.subplots(nrows=nrows, ncols=ncols)

        j = 0
        for env_name in env_names:
            j += 1
            plt.subplot(nrows, ncols, j)
            plt.title(env_name)
            for predictor_name in predictors:
                for agent_name in agent_names:
                    for c in cs:
                        filename = tools.FileNaming.resultName(
                            predictor_name, env_name, agent_name, c
                        )
                        r = Result(filename=filename)
                        n = self.listNorm(r)
                        label = predictor_name + " - "
                        label += agent_name + " - " + str(c)
                        label += " - " + str(int(sum(r.time))) + "s"
                        plt.hist(n, bins, alpha=0.5, label=label)
                        plt.legend(loc='upper right')
        mng = plt.get_current_fig_manager()
        mng.full_screen_toggle()
        plt.show()

    def plotSigma(self, env_name, agent_name="random", c=10000):
        filename = tools.FileNaming.resultName(
            "GP", env_name, agent_name, c
        )
        r = Result(filename=filename)
        n = self.listNorm(r)
        X = []
        Y = []
        for i in range(len(n)):
            X.append(n[i])
            Y.append(r.sigma[i])

        plt.scatter(X, Y)
        the_x = sorted(X)
        plt.plot(the_x, the_x)
        plt.show()
