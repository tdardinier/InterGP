import numpy as np


class Predictor():

    def __init__(self, n, m):
        self.n = n
        self.m = m
        self.std = False

        self.data_X = []
        self.data_U = []
        self.data_Y = []

        self.name = "predictor"

        def f(x): x
        self.normalizer = f

        self.clear()

    def updateNormalizer(self):
        s = self.n + self.m
        m = np.empty([1, s])
        d = np.empty([1, s])
        for i in range(self.n):
            les_x = [x[i] for x in self.data_X]
            # print("les_x", les_x)
            m.put(i, np.average(les_x))
            d.put(i, 1. / np.std(les_x))
        for i in range(self.m):
            les_u = [u[i] for u in self.data_U]
            # print("les_u", les_x)
            m.put(self.n + i, np.average(les_u))
            d.put(self.n + i, 1. / np.std(les_u))

        def f(x):
            return d * (x - m)

        # print("m", m)
        # print("d", d)

        self.normalizer = f

    # def normalize(self, xu):
        # return [self.normalizer(e) for e in xu]

    def clear(self):
        self.data_X = []
        self.data_U = []
        self.data_Y = []

    def addData(self, X, U, Y):
        self.data_X += X
        self.data_U += U
        self.data_Y += Y

    def addElement(self, x, u, y):
        self.data_X.append(x)
        self.data_U.append(u)
        self.data_Y.append(y)

    def train(self):
        print("Not implemented yet")

    def predict(self, x, u):
        print("Not implemented yet")
