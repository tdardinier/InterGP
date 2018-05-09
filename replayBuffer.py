import numpy as np
import math


class ReplayBuffer():

    def __init__(self, x=[], u=[], y=[], filename=None):

        self.x = x
        self.u = u
        self.y = y

        if filename is not None:
            self.load(filename)

    def getFilename(self, name, useSuffix=True):
        prefix = "files/"
        if useSuffix:
            name += ".npy"
        return prefix + name

    def addData(self, x, u, y):
        self.x.append(x)
        self.u.append(u)
        self.y.append(y)

    def save(self, name="undefined"):
        a = np.array([self.x, self.u, self.y])
        np.save(self.getFilename(name), a)

    def load(self, name="undefined"):
        a = np.load(self.getFilename(name, True))
        self.x = list(a[0])
        self.u = list(a[1])
        self.y = list(a[2])

    def slice(self, l):
        (x, u, y) = []
        for (a, b) in l:
            r = range(a, b)
            x += [self.x[i] for i in r]
            u += [self.u[i] for i in r]
            y += [self.y[i] for i in r]
        return ReplayBuffer(x=x, u=u, y=y)

    def crossValidation(self, k, i):
        n = len(self.x)
        size = math.ceil(float(n) / float(k))
        a = size * i
        b = size * (i + 1)
        return (ReplayBuffer([(0, a), (b, n)]), ReplayBuffer(a, b))
