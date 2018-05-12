import numpy as np
import math
import random as rd


class ReplayBuffer():

    def __init__(self, x=[], u=[], y=[], filename=None):

        self.x = x
        self.u = u
        self.y = y

        if filename is not None:
            self.load(filename)

    def shuffle(self, seed=42):
        rd.seed(seed)
        indices = [i for i in range(len(self.x))]
        rd.shuffle(indices)
        x = [self.x[i] for i in indices]
        u = [self.u[i] for i in indices]
        y = [self.y[i] for i in indices]
        return ReplayBuffer(x=x, u=u, y=y)

    def cut(self, n):
        x = self.x[-n:]
        u = self.u[-n:]
        y = self.y[-n:]
        return ReplayBuffer(x=x, u=u, y=y)

    def addData(self, x, u, y):
        self.x.append(x)
        self.u.append(u)
        self.y.append(y)

    def save(self, filename):
        a = np.array([self.x, self.u, self.y])
        print("ReplayBuffer: Saving " + filename + "...")
        np.save(filename, a)
        print("ReplayBuffer: Saved!")

    def load(self, filename):
        print("ReplayBuffer: Loading " + filename + "...")
        a = np.load(filename)
        print("ReplayBuffer: Loaded!")
        self.x = list(a[0])
        self.u = list(a[1])
        self.y = list(a[2])

    def slice(self, l):
        (x, u, y) = ([], [], [])
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
        train = self.slice([(0, a), (b, n)])
        test = self.slice([(a, b)])
        return (train, test)
