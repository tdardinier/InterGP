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


class InterGP:

    def __init__(self, k, n=1):
        self.k = k  # kernel function
        self.n = n

        self.X = None
        self.Y = None

    def generateMatrixCov(self, X1, X2):
        m = []
        for x1 in X1:
            line = []
            for x2 in X2:
                line.append(self.k(x1, x2))
            m.append(line)
        return np.matrix(m)

    def fit(self, X, Y):
        # X = [X[0], ..., X[N-1]] -> dim n (array even if n == 1)
        # Y = [Y[0], ..., Y[N-1]] -> scalars
        assert (len(X) == len(Y)), "Wrong sizes X Y"
        self.X = X
        self.Y = Y
        N = len(self.Y)
        f = np.matrix(Y).reshape((N, 1))
        K = self.generateMatrixCov(X, X)  # K(X, X)
        self.inv_K = np.linalg.inv(K)
        self.inv_K_f = self.inv_K * f

    def predict(self, x):
        K_star = self.generateMatrixCov([x], self.X)
        k_double_star = self.k(x, x)
        mean = K_star * self.inv_K_f
        var = k_double_star - K_star * self.inv_K * (K_star.T)
        return (mean, var)


def f(x):
    return x[0] * x[1]


X = [[np.random.rand() * 10, np.random.rand() * 10] for i in range(100)]
Y = [f(x) for x in X]

gp = InterGP(k)
gp.fit(X, Y)
