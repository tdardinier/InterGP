import numpy as np
from interval_matrix import InterMatrix
from interval import Interval as In
from function import square, exp


# x : n * 1, y : n * 1
# exponential
def k(x, y):
    s = 0.0
    assert (len(x) == len(y)), "Not same length"
    n = len(x)
    for i in range(n):
        s += (x[i] - y[i]) ** 2
    return np.exp(-0.5 * s)


def ik(x, y):
    s = In(0.0)
    assert (len(x) == len(y)), "Not same length"
    n = len(x)
    for i in range(n):
        diff = In.add(x[i], In.neg(y[i]))
        sq = square.image(diff)
        s = In.add(s, sq)
    s = In.mult(In(-0.5), s)
    return exp.image(s)


class InterGP:

    def __init__(self, k, ik, n=1):
        self.k = k  # kernel function
        self.ik = ik  # same function but on interval
        self.n = n

        self.X = None
        self.Y = None

    def generateMatrixCov(self, X1, X2, inter=False):
        m = []
        for x1 in X1:
            line = []
            for x2 in X2:
                if inter:
                    line.append(self.ik(x1, x2))
                else:
                    line.append(self.k(x1, x2))
            m.append(line)
        if inter:
            return InterMatrix(m)
        else:
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

        interX = [[In(yy) for yy in y] for y in self.X]

        K_star = self.generateMatrixCov([x], interX, True)
        print("K star", K_star)
        k_double_star = self.ik(x, x)
        print("K double star", k_double_star)

        interInvK = InterMatrix.createFromMatrix(self.inv_K)
        interInvKf = InterMatrix.createFromMatrix(self.inv_K_f)

        print("interInvKf", interInvKf)

        mean = InterMatrix.mult(K_star, interInvKf)

        K_star_T = K_star.transpose()
        prod = InterMatrix.mult(interInvK, K_star_T)
        prod = InterMatrix.mult(K_star, prod)
        prod = prod.neg()

        var = In.add(k_double_star, prod.matrix[0][0])

        # var = k_double_star - K_star * self.inv_K * (K_star.T)

        return (mean, var)


def f(x):
    return x[0] * x[1]


def test(n=50, noise=0.001, x=3, y=4):
    X = [[np.random.rand() * 10, np.random.rand() * 10] for i in range(n)]
    Y = [f(a) for a in X]

    gp = InterGP(k, ik)
    gp.fit(X, Y)
    xx = [In(x), In(y)]
    x = [In.fromNoise(x, noise), In.fromNoise(y, noise)]
    a = gp.predict(xx)
    b = gp.predict(x)
    print("\nRESULTS:")
    print(a)
    print(b)
