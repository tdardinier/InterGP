import math
import random as rd

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
