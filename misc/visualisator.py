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
                plt.fill_between(x, prev_y1, y1[j], alpha=0.5, color=color,
                                 label="Confidence interval at " +
                                 str(int(p[j] * 100)) + "%")
                plt.fill_between(x, prev_y2, y2[j], alpha=0.5, color=color)
                prev_y1, prev_y2 = y1[j], y2[j]

            plt.plot(x, y, label="Estimated state")
            plt.scatter(x, ry, label="Real state", color="black")

            if ii == 0:
                plt.legend(loc=loc)
        self.__fullScreen()

    def plotCompGP(self,
                   traj,
                   color='#0088FF',
                   name="Test",
                   components=None,
                   loc="upper left",
                   ):

        if components is None:
            components = list(range(len(traj.X[0])))

        n_components = len(components) + 1

        plt.subplots(nrows=n_components, ncols=1)
        for ii in range(len(components)):
            i = components[ii]
            plt.subplot(n_components, 1, ii+1)
            s = name
            s += ", dimension " + str(i)
            plt.title(s)
            x = []
            y = []
            y1 = []
            y2 = []

            for t in range(len(traj.S)):
                x.append(t)
                y.append(traj.X[t].item(i))
                y1.append(traj.S[t][i][0])
                y2.append(traj.S[t][i][1])

            # color = 'blue'
            plt.fill_between(x, y1, y2, color=color, label="Approximation")
            plt.plot(x, y, label="Real state", color='black')

            if ii == 0:
                plt.legend(loc=loc)

        plt.subplot(n_components, 1, n_components)
        plt.fill_between(x, traj.P, label='Probability')

        self.__fullScreen()
