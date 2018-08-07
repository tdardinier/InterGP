from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
import numpy.random as rd

rd.seed(42)
kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(1, (1e-2, 1e2))
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)

def f(x):
    return x[0] * x[1]

n = 20
X = [[rd.random(), rd.random()] for _ in range(n)]
Y = [f(x) for x in X]
gp.fit(X, Y)

x = [[0.2, 0.2]]
print(gp.predict(x, return_std=True))
print(gp.predict(x, return_cov=True))
