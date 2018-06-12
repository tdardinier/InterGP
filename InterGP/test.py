from interGP import InterGP
from interGPPlus import InterGPPlus as plus
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


def f(x):
    return x[0] * x[1]


def testInterGP():

    X = [[np.random.rand() * 10, np.random.rand() * 10] for i in range(100)]
    Y = [f(x) for x in X]

    gp = InterGP(k)
    gp.fit(X, Y)
    print(gp.predictState([(1.5, 2.5), (2.5, 3.5)]))


def testInterGPPlus(x=[3, 3]):

    X = [[np.random.rand() * 10, np.random.rand() * 10] for i in range(100)]
    Y = [f(xx) for xx in X]
    YY = [[2*y, y-1] for y in Y]

    p_dep = plus(k, 2, dependent=True, sigma=[1, 0])
    p_dep.fit(X, YY)

    p_inde = plus(k, 2)
    p_inde.fit(X, YY)

    print("DEP", p_dep.predictSingle(x))
    print("INDE", p_inde.predictSingle(x))

    print("INDE state", p_inde.predictState([[2.9, 3.1], [3.9, 4.1]]))


def test(n=50, noise=0.1, x=3, y=4, p=0.95):
    X = [[np.random.rand() * 10, np.random.rand() * 10] for i in range(n)]
    Y = [f(a) for a in X]
    YY = [[2*yy, yy-1] for yy in Y]
    p_inde = plus(k, 2)
    p_inde.fit(X, YY)
    xx = [(x - noise, x + noise), (y - noise, y + noise)]
    print("\nRESULTS:", xx)
    print(p_inde.predictState(xx))
    print("COMPARED TO", [2 * x * y, x * y - 1])
