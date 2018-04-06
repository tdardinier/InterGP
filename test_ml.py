import numpy as np
import random

def genPoint():
    A = np.matrix([[2, 0], [0, -2]])
    B = np.matrix([[10], [10]])
    X = np.matrix([[random.random()], [random.random()]])
    return (X, A * X + B)

n = 10

x = np.zeros([2, 0])
y = np.zeros([2, 0])

for i in range(n):
    (X, Y) = genPoint()
    y = np.hstack([y, Y])
    x = np.hstack([x, X])

x = np.vstack([x, np.ones(n)])
x = x.T
y = y.T

p = np.linalg.lstsq(x, y)[0]
print(p)
