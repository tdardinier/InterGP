import predictor
import numpy as np
# from matplotlib import pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C


class Predictor(predictor.Predictor):

    def __init__(self, n=4, m=1):
        super().__init__(n, m)
        self.std = True

    def formatInput(self, x, u):
        xx = np.array(x)
        uu = np.array(u)
        x = np.vstack([xx, uu]).T
        # print("---------------------------------")
        # print("---------------------------------")
        # print(x)
        # print("---------------------------------")
        # print(self.normalizer(x))
        # print("---------------------------------")
        # print("---------------------------------")
        return self.normalizer(x)

    def getFormattedInput(self, X, U):
        R = np.empty([0, self.n + self.m])
        for x, u in zip(X, U):
            c = self.formatInput(x, u)
            R = np.vstack([R, c])
        return R

    def train(self):

        self.updateNormalizer()

        # Instanciate a Gaussian Process model
        self.kernel = [
            C(1.0, (1e-3, 1e3)) * RBF(1, (1e-2, 1e2))
            for _ in range(self.n)]
        self.gp = [GaussianProcessRegressor(
            kernel=self.kernel[i],
            n_restarts_optimizer=9
        )
            for i in range(self.n)]

        X = self.getFormattedInput(self.data_X, self.data_U)
        for i in range(self.n):
            y = np.reshape([v[i] for v in self.data_Y], [-1, 1])
            print(np.shape(X), np.shape(y))
            print(self.data_X[0])
            print(self.data_U[0])
            print(X[0])
            self.gp[i].fit(X, y)
            print("Done", i)

    def predict(self, xx, uu, return_std=False):
        x = self.formatInput(xx, uu)
        y = np.empty([self.n, 1])
        sigma = np.empty([self.n, 1])
        for i in range(self.n):
            xy, xsigma = self.gp[i].predict(x, return_std=True)
            y.put(i, xy)
            sigma.put(i, xsigma)
        return (y, sigma)
