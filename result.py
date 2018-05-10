import numpy as np


class Result():

    def __init__(self, k=10, c=1000, filename=None):
        self.k = k
        self.c = c
        self.x = []
        self.u = []
        self.real_y = []
        self.predicted_y = []
        self.time = []
        self.sigma = []

        if filename is not None:
            self.load(filename=filename)

    def addResults(self, x, u, real_y, predicted_y, time, sigma=None):
        self.x.append(x)
        self.u.append(u)
        self.real_y.append(real_y)
        self.predicted_y.append(predicted_y)
        self.time.append(time)
        if sigma is not None:
            self.sigma.append(sigma)

    def getFilename(self, name, useSuffix=True):
        prefix = "results/"
        if useSuffix:
            name += ".npy"
        return prefix + name

    def save(self, filename="Undefined"):
        l = []
        l.append(self.k)
        l.append(self.c)
        l.append(self.x)
        l.append(self.u)
        l.append(self.real_y)
        l.append(self.predicted_y)
        l.append(self.time)
        l.append(self.sigma)
        a = np.array(l)
        np.save(self.getFilename(filename), a)

    def load(self, filename="undefined"):
        name = self.getFilename(filename, True)
        a = np.load(name)
        print("Loading " + name + "...")
        self.k = int(a[0])
        self.c = int(a[1])
        self.x = list(a[2])
        self.u = list(a[3])
        self.real_y = list(a[4])
        self.predicted_y = list(a[5])
        self.time = list(a[6])
        self.sigma = list(a[7])
