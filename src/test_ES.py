import systident.entropySearch as ES
import numpy.random as rd
import math as m


def f1(x):
    return (x[0] - 2) ** 2 * (x[1] - 3) ** 2


bounds1 = [[0, 5], [0, 5]]


def f2(x):
    xx = x[0]
    return m.sin(xx) + 0.1 * (xx-2) ** 2


bounds2 = [[-3, 6]]


def findMinimum(f, bounds):

    d = len(bounds)
    es = ES.EntropySearch(d=d, bounds=bounds)

    initial_k = 5
    for _ in range(initial_k):
        x = []
        for j in range(d):
            a, b = bounds[j]
            x.append((b - a) * rd.random() + a)
        es.addObs(x, f(x))

    current_prob = .0
    while current_prob < 0.9:
        x, c_min, current_prob, _ = es.iterate()
        es.addObs(x, f(x))

    print("Minimum", c_min)

    return es
