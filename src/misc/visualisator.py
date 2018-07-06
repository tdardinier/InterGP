import numpy as np
from misc.result import Result
# from replayBuffer import ReplayBuffer
from misc import tools
import matplotlib.pyplot as plt
import math
from scipy.stats import norm
import os


class Visualisator():

    def __init__(self, show=True):
        self.show = show

    def __normL(self, y, yy, norm=2):
        delta = y - yy
        return np.linalg.norm(delta, norm)

    def __scale(self):
        mng = plt.get_current_fig_manager()
        mng.full_screen_toggle()
        if not self.show:
            figure = plt.gcf()
            figure.set_size_inches(25, 14)

    def __fullScreen(self, filename="untitled"):
        self.__scale()
        if self.show:
            print("Showing...")
            plt.show()
        else:
            print("Saving to " + filename + "...")
            directory = os.path.dirname(filename)
            if not os.path.exists(directory):
                os.makedirs(directory)
            plt.savefig(filename, dpi=100)
            print("Saved!")

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
                   trajs,
                   colors=['#0088FF', 'blue'],
                   name="Test",
                   components=None,
                   loc="upper left",
                   k=10,
                   filename='untitled',
                   n_cols=2,
                   ):

        traj = trajs[0]

        if components is None:
            components = list(range(len(traj.X[0])))

        kk = min(len(traj.S), k + 1)
        n_components = len(components)
        n_rows = math.ceil(n_components / n_cols)

        plt.subplots(nrows=n_rows, ncols=n_cols)
        plt.title(name)
        for index in range(len(trajs)):
            traj = trajs[index]
            color = colors[index]
            for ii in range(len(components)):
                i = components[ii]
                plt.subplot(n_rows, n_cols, ii+1)
                if ii == 0:
                    plt.title(name + "\nDimension " + str(i))
                else:
                    plt.title("Dimension " + str(i))
                x = []
                y = []
                y1 = []
                y2 = []

                for t in range(kk):
                    x.append(t)
                    y.append(traj.X[t].item(i))
                    y1.append(traj.S[t][i][0])
                    y2.append(traj.S[t][i][1])

                # color = 'blue'
                plt.fill_between(x, y1, y2, color=color,
                                 label="Approximation", alpha=0.5)
                plt.plot(x, y, label="Real state", color='black')

                M = max(y)
                m = min(y)
                delta = max(M - m, 0.01) * 0.02

                for j in range(kk - 1):

                    a = 0.2
                    b = 1 - a
                    xx = j + a
                    yy = a * y[j] + b * y[j + 1]
                    if y[j+1] < y[j]:
                        yy = b * y[j] + a * y[j + 1]
                    yy += delta
                    # yy = 0.5 * (y[j] + y[j+1])

                    rp = "{:3.3f}".format(traj.realP[j][i])
                    # ap = "{:5.3f}".format(traj.aimedP[j][i])
                    # s = rp+" ("+ap+")"
                    s = rp
                    if s[0] == '0':
                        s = s[1:]

                    plt.text(xx, yy, s, {}, rotation=0)

                if ii == 0:
                    plt.legend(loc=loc)

        # plt.subplot(n_components, 1, n_components)
        # plt.fill_between(x, traj.P[:kk], label='Probability')

        self.__fullScreen(filename=filename)
