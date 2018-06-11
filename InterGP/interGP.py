import numpy as np
from scipy.optimize import fmin_tnc


class InterGP:

    def __init__(self, k, n=1):

        self.k = k  # kernel function

        self.X = None
        self.Y = None
        self.N = None

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
        self.N = len(self.Y)
        f = np.matrix(Y).reshape((self.N, 1))
        K = self.generateMatrixCov(X, X)  # K(X, X)
        self.B = np.linalg.inv(K)
        self.A = self.B * f

    def predictSingle(self, x):
        K_star = self.generateMatrixCov([x], self.X)
        k_double_star = self.k(x, x)
        mean = K_star * self.A
        var = k_double_star - K_star * self.B * (K_star.T)
        return (mean, var)

    def predictState(self, bounds, p=0.95):
        # TODO p -> alpha
        alpha = 1.96

        def mu(x):
            K_star = self.generateMatrixCov([x], self.X)
            s = 0.0
            for i in range(self.N):
                s += K_star[0, i] * self.A[i, 0]
            return s

        def delta(x):
            K_star = self.generateMatrixCov([x], self.X)
            s = self.k(x, x)
            for i in range(self.N):
                for j in range(self.N):
                    s -= K_star[0, i] * K_star[0, j] * self.B[i, j]
            return s

        def m(x):
            return mu(x) - alpha * delta(x)

        def M(x):
            return mu(x) + alpha * delta(x)

        def minusM(x):
            return - M(x)

        x0 = np.array([0.5 * (x[0] + x[1]) for x in bounds])
        xa = fmin_tnc(m, x0, approx_grad=True, bounds=bounds)
        xb = fmin_tnc(minusM, x0, approx_grad=True, bounds=bounds)

        print(xa, xb)

        return (m(xa[0]), M(xb[0]))


def f(x):
    return x[0] * x[1]


# x : n * 1, y : n * 1
# exponential
def k(x, y):
    s = 0.0
    assert (len(x) == len(y)), "Not same length"
    n = len(x)
    for i in range(n):
        s += (x[i] - y[i]) ** 2
    return np.exp(-0.5 * s)


X = [[np.random.rand() * 10, np.random.rand() * 10] for i in range(100)]
Y = [f(x) for x in X]

gp = InterGP(k)
gp.fit(X, Y)
