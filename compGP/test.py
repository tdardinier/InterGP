from gp import GP
import numpy as np


# x : n * 1, y : n * 1
# exponential
def k(x, y):
    s = 0.0
    assert (len(x) == len(y)), "Not same length"
    n = len(x)
    for i in range(n):
        s += (x[i] - y[i]) ** 2
    return np.exp(-0.5 * s)


def trainGP(n=100):

    def f(x):
        return x[0] * x[1]

    X = [[np.random.rand() * 10, np.random.rand() * 10] for i in range(100)]
    Y = [f(xx) for xx in X]

    gp = GP(k, 0, n=2, m=0)
    gp.fit(X, Y)

    return gp


def testSmallGP(x=2, y=2, noise=0.1, inter=[3.5, 4.5]):
    gp = trainGP()
    return gp.computePik([[(x - noise, x + noise), (y - noise, y + noise)]], inter)


def testCompGP(x=2, y=2, noise=0.01, inter=[3.8, 4.2]):
    gp = trainGP()
    S_0 = [(1, 1), (2, 2)]
    S_1 = [(2 - noise, 2 + noise), (1 - noise, 1 + noise)]
    S_2 = [(x - noise, x + noise), (y - noise, y + noise)]
    return gp.computePik([S_0, S_1, S_2], inter)


print(testSmallGP())
