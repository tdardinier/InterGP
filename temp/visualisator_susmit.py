import numpy as np
from misc.result import Result
# from replayBuffer import ReplayBuffer
from misc import tools
import matplotlib.pyplot as plt
import math
from scipy.stats import norm


class Visualisator():

    def __init__(self):
        pass

    def __normL(self, y, yy, norm=2):
        delta = y - yy
        return np.linalg.norm(delta, norm)

    def __fullScreen(self):
        print("Showing...")
        mng = plt.get_current_fig_manager()
        mng.full_screen_toggle()
        plt.show()

    def listNorm(self, r):
        n = []
        for i in range(len(r.real_y)):
            ry = r.real_y[i]
            py = r.predicted_y[i]
            n.append(self.__normL(ry, py, 1) / r.n)
        return n

    def compare(self,
                predictors, envs, agent_names,
                cs, norm=1, density=False):
        print(density)
        nrows = 1
        ncols = 1
        n = len(envs)
        if n > 1:
            ncols = 2
        if n > 8:
            ncols = 3
        nrows = math.ceil(n / ncols)

        plt.subplots(nrows=nrows, ncols=ncols)

        j = 0
        for env in envs:
            labels = []
            data = []
            for agent_name in agent_names:
                j += 1
                plt.subplot(nrows, ncols, j)
                s = env.name + " (" + str(env.n)
                s += ", " + str(env.m) + ")"
                s += " - " + agent_name
                plt.title(s)
                for predictor_name in predictors:
                    for c in cs:
                        filename = tools.FileNaming.resultName(
                            predictor_name, env.name, agent_name, c
                        )
                        r = Result(filename=filename)
                        n = self.listNorm(r)
                        label = predictor_name
                        label += " - " + str(c)
                        label += " - " + str(int(sum(r.time))) + " s"
                        data.append(n)
                        labels.append(label)
            plt.hist(data, 100, label=labels, density=density)
            plt.legend(loc='upper right')
        self.__fullScreen()

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
        self.__fullScreen()

    def plotDimensions(self,
                       predictor_name="NGP",
                       env_name="CartPole-v1",
                       agent_name="random",
                       c=100,
                       components=[0],
                       p=[0.5, 0.7, 0.9, 0.99],
                       size=50,
                       loc="upper left",
                       ):

        filename = tools.FileNaming.resultName(
            predictor_name, env_name, agent_name, c
        )
        r = Result(filename=filename)
        plt.subplots(nrows=len(components), ncols=1)
        for ii in range(len(components)):
            i = components[ii]
            plt.subplot(len(components), 1, ii+1)
            s = env_name
            s += ", dimension " + str(i)
            s += ", agent: " + agent_name
            s += ", c = " + str(c)
            s += ", p = " + str(p)
            plt.title(s)
            x = []
            y = []
            ry = []
            y1 = [[] for _ in p]
            y2 = [[] for _ in p]
            alphas = [norm.ppf(0.5 * (1. + pp)) for pp in p]

            for t in range(size):
                x.append(t)
                yy = r.predicted_y[t, i]
                sigma = r.sigma[t, i]
                y.append(yy)
                for j in range(len(alphas)):
                    alpha = alphas[j]
                    y1[j].append(yy - alpha * sigma)
                    y2[j].append(yy + alpha * sigma)
                ry.append(r.real_y[t, i])

                if t in [20, 40] and predictor_name == "FNGP":
                    plt.axvline(x=t)

            cmap = plt.get_cmap('jet_r')
            prev_y1 = y
            prev_y2 = y
            for j in range(len(p)):
                color = cmap(float(j) / len(p))

                aa = 0
                bb = 21
                plt.fill_between(x[aa:bb], prev_y1[aa:bb], y1[j][aa:bb], alpha=0.5, color=color,
                                 label="Confidence interval at " +
                                 str(int(p[j] * 100)) + "%")

                aa = 21
                bb = 41
                plt.fill_between(x[aa:bb], prev_y1[aa:bb], y1[j][aa:bb], alpha=0.5, color=color)

                aa = 41
                bb = 100
                plt.fill_between(x[aa:bb], prev_y1[aa:bb], y1[j][aa:bb], alpha=0.5, color=color)

                aa = 0
                bb = 21
                plt.fill_between(x[aa:bb], prev_y2[aa:bb], y2[j][aa:bb], alpha=0.5, color=color)

                aa = 21
                bb = 41
                plt.fill_between(x[aa:bb], prev_y2[aa:bb], y2[j][aa:bb], alpha=0.5, color=color)

                aa = 41
                bb = 100
                plt.fill_between(x[aa:bb], prev_y2[aa:bb], y2[j][aa:bb], alpha=0.5, color=color)




                prev_y1, prev_y2 = y1[j], y2[j]

            plt.plot(x[:21], y[:21], label="Estimated state", color="#333333")
            plt.plot(x[21:41], y[21:41], color="#333333")
            plt.plot(x[41:], y[41:], color="#333333")
            plt.scatter(x, ry, label="Real state", color="black")

            if ii == 0:
                plt.legend(loc=loc)
        self.__fullScreen()
