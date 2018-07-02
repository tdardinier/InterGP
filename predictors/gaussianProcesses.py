from misc import predictor
import numpy as np
# from matplotlib import pyplot as plt
from misc.coreGP import CoreGP


class Predictor(predictor.Predictor):

    def __init__(self, n=4, m=1):
        super().__init__(n, m)
        self.name = "GP"
        self.std = True
        self.separated = True
        self.n = n

    def getDistance(self, xu1, xu2):
        return np.linalg.norm(xu1 - xu2)

    def getCouples(self):
        X = [(np.vstack([self.data_X[i], self.data_U[i]]),
              self.data_Y[i]) for i in range(len(self.data_X))]
        return X

    def getTriples(self):
        X = [(self.data_X[i], self.data_U[i],
              self.data_Y[i]) for i in range(len(self.data_X))]
        return X

    def getClosestNeighbours(self, x, n=100):
        X = self.getTriples()
        X.sort(key=lambda e:
               self.getDistance(x, e[0]))
        return X[:n]

    def getClosestNeighboursXU(self, xu, n=100):
        X = self.getCouples()
        X.sort(key=lambda e:
               self.getDistance(xu, e[0]))
        return X[:n]

    def getXY(self):
        X = np.empty([0, self.n + self.m])
        for (x, u) in zip(self.data_X, self.data_U):
            xu = np.vstack([x, u]).T
            X = np.vstack([X, xu])
        Y = np.empty([0, self.n])
        for y in self.data_Y:
            Y = np.vstack([Y, y.T])
        return X, Y

    def train(self):
        (X, Y) = self.getXY()
        if self.separated:
            self.gp = []
            for i in range(self.n):
                # self.gp.append(tools.GaussianProcesses())
                self.gp.append(CoreGP())
                y = np.matrix([[Y[k, i]] for k in range(len(Y))])
                self.gp[i].train(X, y)
        else:
            # self.gp = tools.GaussianProcesses()
            self.gp = CoreGP()
            self.gp.train(X, Y)

    def concat(self, XU):
        A = np.empty([0, np.shape(XU)[2]])
        for xu in XU:
            A = np.vstack([A, xu])
        return A

    def extractXU(self, x):
        xx = x[0]
        uu = x[1]
        xu = np.vstack([xx, uu]).T
        return xu

    def predict(self, xx, uu):
        xu = np.vstack([xx, uu]).T
        # if self.each_time:
        # XUY = self.getClosestNeighboursXU(xu)
        # XUY = self.getClosestNeighbours(xx.T)
        # XU = self.concat([self.extractXU(x) for x in XUY])
        # Y = self.concat([x[2].T for x in XUY])
        # gp = tools.GaussianProcesses()
        # gp.train(XU, Y)
        # y, sigma = gp.predict(xu)
        if self.separated:
            y = []
            sigma = []
            for pred in self.gp:
                yy, ssigma = pred.predict(xu)
                y.append([yy.item(0)])
                sigma.append(ssigma.item(0))
            y = np.matrix(y)
            sigma = np.matrix([sigma])
        else:
            y, sigma = self.gp.predict(xu)
            y = y.T
        if self.std:
            return y, sigma
        return y
