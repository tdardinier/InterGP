import math
import random as rd
import scipy.linalg
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C


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
        # print(self.B)
        # print(P)
        # print(b)
        inv_b = scipy.linalg.inv(b)
        new_P -= a * inv_b * c
        new_P += self.Q
        return new_P

    def solve(self, x):
        current_P = self.Q
        for i in range(self.N):
            current_P = self.computeP(current_P)
            self.P.append(current_P)
        F = scipy.linalg.inv(self.B.T * current_P * self.B) * (self.B.T * current_P * self.A)
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
        return self.mini[i] + (y + 0.5) * (self.maxi[i] - self.mini[i]) / self.n_div

    def unquantizeRandom(self, y, i):
        return self.mini[i] + (y + rd.random()) * (self.maxi[i] - self.mini[i]) / self.n_div

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


def proba(v, tau):
    return math.exp(v / tau)


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


class GaussianProcesses():

    def __init__(self):
        pass

    def getData(self, dim=2, n=20):

        def f(x):
            m = 1
            for xx in x:
                m *= xx
            return m

        def getX():
            return [(rd.random() - 0.5) * 5 for _ in range(dim)]

        X = [getX() for _ in range(n)]
        Y = [f(x) for x in X]

        return (X, Y)

    def train(self, X, Y):
        self.kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
        self.gp = GaussianProcessRegressor(kernel=self.kernel,
                                           n_restarts_optimizer=9)
        self.gp.fit(X, Y)

    def predict(self, x):
        y, sigma = self.gp.predict(x, return_std=True)
        return y, sigma
