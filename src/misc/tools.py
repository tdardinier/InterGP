import math
import random as rd
import scipy.linalg
import numpy as np


class LQR():

    def update(self, A, B, Q, N):
        self.A = A
        self.B = B
        self.Q = Q
        self.N = N

    def __init__(self, A, B, Q, N):
        self.update(A, B, Q, N)

        self.P = []

    def computeP(self, P):
        new_P = self.A.T * P * self.A
        a = self.A.T * P * self.B
        b = self.B.T * P * self.B
        c = self.B.T * P * self.A
        inv_b = scipy.linalg.inv(b)
        new_P -= a * inv_b * c
        new_P += self.Q
        return new_P

    def solve(self, x):
        current_P = self.Q
        for i in range(self.N):
            current_P = self.computeP(current_P)
            self.P.append(current_P)
        F = scipy.linalg.inv(self.B.T * current_P * self.B) * \
            (self.B.T * current_P * self.A)
        return - F * x


class Quantizer():

    def __init__(self, n_div, mini, maxi):
        self.n_div = float(n_div)
        self.maxi = [float(x) for x in maxi]
        self.mini = [float(x) for x in mini]

    def quantize(self, x, i):
        y = (x - self.mini[i]) / (self.maxi[i] - self.mini[i])
        y = min(0.99, max(0, y))
        return int(y * self.n_div)

    def unquantize(self, y, i):
        return self.mini[i] + \
            (y + 0.5) * (self.maxi[i] - self.mini[i]) / self.n_div

    def unquantizeRandom(self, y, i):
        return self.mini[i] + \
            (y + rd.random()) * (self.maxi[i] - self.mini[i]) / self.n_div

    def undiscretizeRandom(self, s):
        x = []
        for i in range(4):
            x.append(self.unquantizeRandom(s % self.n_div, i))
            s = s // self.n_div
        return x

    def undiscretize(self, s):
        x = []
        for i in range(4):
            x.append(self.unquantize(s % self.n_div, i))
            s = s // self.n_div
        return x

    def discretize(self, obs):
        s = 0
        m = 1
        for i in range(4):
            x = obs[i]
            s += m * self.quantize(x, i)
            m *= self.n_div
        return int(s)


class FileNaming():

    @staticmethod
    def __assembly(folder, liste, suffix):
        return "data/" + folder + "_".join(liste) + suffix

    @staticmethod
    def trajName(env_name, agent_name, c, p, conf):
        folder = "trajs" + "/"
        suffix = ".npy"
        liste = [env_name, agent_name, str(c), str(p)]
        liste += conf.getListe()
        return FileNaming.__assembly(folder, liste, suffix)

    @staticmethod
    def replayName(env_name, agent_name):
        folder = "replays" + "/"
        suffix = ".npy"
        return FileNaming.__assembly(folder, [env_name, agent_name], suffix)

    @staticmethod
    def resultName(predictor_name, env_name, agent_name, c):
        folder = "results" + "/"
        suffix = ".npz"
        liste = [predictor_name, env_name, agent_name, str(c)]
        return FileNaming.__assembly(folder, liste, suffix)

    @staticmethod
    def modelName(env_wrapper):
        folder = "models" + "/"
        suffix = ".pkl"
        return FileNaming.__assembly(folder, [env_wrapper.name], suffix)


class Normalizer():

    def __init__(self, data):
        epsilon = 0.0000000000001
        n = np.shape(data)[1]
        print("Tools Normalizer: n", n)
        m = np.empty([n, 1])
        d = np.empty([n, 1])
        for i in range(n):
            elements = [x[i] for x in data]
            m.put(i, np.average(elements))
            dev = np.std(elements)
            if dev < epsilon:
                d.put(i, 0)
            else:
                d.put(i, 1. / dev)

        def f(x):
            return np.multiply(d, (x - m))

        self.f = f

    def normalize(self, data):
        return [self.f(x) for x in data]


def proba(v, tau):
    return math.exp(v / tau)


def less_than(x, M):
    if M is None:
        return True
    return x < M


def softmax(values, tau):
    p = [proba(x, tau) for x in values]
    seuil = rd.random() * sum(p)
    s = 0.0
    i = 0
    while s < seuil:
        s += p[i]
        i += 1
    return i - 1


def getMax(threshold, n):
    add = threshold / (n // 2 - 1)
    return threshold + add


def getM(env):
    m = 1
    for x in env.action_space.shape:
        m *= x
    return m
