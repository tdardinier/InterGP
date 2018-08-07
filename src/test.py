import misc.coreGP as CGP
import numpy as np


def f(x): return x[0] * x[1]


n = 20
X = [[np.random.random(), np.random.random()] for _ in range(20)]
Y = [f(x) for x in X]

gp = CGP.CoreGP()
gp.train(X, Y)

x1 = [0.5, 0.2]
x2 = [0.5, 0.3]
x3 = [0.5, 0.4]

xs = [x1, x2, x3]
