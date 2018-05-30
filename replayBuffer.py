import numpy as np
import math
import random as rd
import tools


class ReplayBuffer():

    def __init__(self, x=[], u=[], y=[], filename=None):

        self.x = list(x)
        self.u = list(u)
        self.y = list(y)

        if filename is not None:
            self.load(filename)

    def getTrajectories(self):
        rs = []
        r = ReplayBuffer()
        prev_y = self.x[0]
        for i in range(len(self.x)):
            x = self.x[i]
            u = self.u[i]
            y = self.y[i]
            if np.array_equal(x, prev_y):
                r.addData(x, u, y)
                prev_y = y
            else:
                rs.append(r)
                if i + 1 < len(self.x):
                    r = ReplayBuffer()
                    prev_y = self.x[i+1]
        return rs

    def normalize(self):
        xy = self.x + self.y
        n_xy = tools.Normalizer(xy)
        x = n_xy.normalize(self.x)
        y = n_xy.normalize(self.y)
        n_u = tools.Normalizer(self.u)
        u = n_u.normalize(self.u)
        return ReplayBuffer(x=x, u=u, y=y)

    def removeX(self):
        y = [self.y[i] - self.x[i] for i in range(len(self.x))]
        return ReplayBuffer(x=self.x, u=self.u, y=y)

    def __str__(self):
        s = "\n\nReplayBuffer\nx: "
        for x in self.x:
            s += str(x) + ","
        s += "\nu: "
        for u in self.u:
            s += str(u) + ","
        s += "\ny: "
        for y in self.y:
            s += str(y) + ","
        s += "\n"
        return s

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
