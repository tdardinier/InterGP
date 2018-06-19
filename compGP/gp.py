import numpy as np
from scipy.optimize import fmin_tnc
from scipy.stats import norm


class GP:

    def __init__(self, k, i, n, m=1):

        self.k = k  # kernel function
        self.i = i  # ith component
        self.n = n  # dimension of state space
        self.m = m  # dimension of action space

        self.X = None
        self.Y = None
        self.N = None

    def __generateMatrixCov(self, X1, X2):
        m = []
        for x1 in X1:
            line = []
            for x2 in X2:
                line.append(self.k(x1, x2))
            m.append(line)
        return np.matrix(m)

    def __probInter(self, mu=0, std=1, inter=[-1.96, 1.96]):
        print("Prob", mu, std, inter)
        return norm.cdf(inter[1], mu, std) - norm.cdf(inter[0], mu, std)

    def __preCompute(self):
        f = np.matrix(self.Y).reshape((self.N, 1))
        K = self.__generateMatrixCov(self.X, self.X)  # K(X, X)
        self.B = np.linalg.inv(K)
        self.A = self.B * f

    def __extractMuSigma(self, xs):
        # xs = [x_0, ..., x_{k-1}]
        # x_i = np.matrix([[], ..., []])
        # x_i is a concatenation of state and action

        k = len(xs)

        K_star = self.__generateMatrixCov(xs, self.X)
        MU = K_star * self.A
        SIGMA = self.__generateMatrixCov(xs, xs) - K_star * self.B * K_star.T

        MU_1 = MU[np.ix_(range(k-1))]
        MU_2 = MU[np.ix_([k-1])]

        SIGMA_11 = SIGMA[np.ix_(range(k-1), range(k-1))]
        SIGMA_12 = SIGMA[np.ix_(range(k-1), [k-1])]
        SIGMA_21 = SIGMA[np.ix_([k-1], range(k-1))]
        SIGMA_22 = SIGMA[np.ix_([k-1], [k-1])]

        prod = SIGMA_21 * np.linalg.inv(SIGMA_11)
        x = np.matrix([xx.item(self.i) for xx in xs[1:]]).T
        mu = MU_2 + prod * (x - MU_1)
        sigma = SIGMA_22 - prod * SIGMA_12

        return mu.item(0), sigma.item(0)

    def __unpack(self, xx):
        print("UNPACK", xx)
        length = self.n + self.m
        k = len(xx) // length
        x = [[xx[i * length + j] for j in range(length)] for i in range(k)]
        return x

    def __createM(self, p=0.95, bigM=True):
        alpha = [norm.ppf(0.5 * (1. + p))]  # array so nonlocal

        if bigM:
            def f(packed_xs):
                # xs = [x_0, ..., x_{k-1}]
                # x_i is a concatenation of the state and the action
                xs = self.__unpack(packed_xs)
                mu, sigma = self.__extractMuSigma(xs)
                return mu - alpha[0] * sigma
        else:
            def f(packed_xs):
                # xs = [x_0, ..., x_{k-1}]
                # x_i is a concatenation of the state and the action
                xs = self.__unpack(packed_xs)
                mu, sigma = self.__extractMuSigma(xs)
                return mu + alpha[0] * sigma

        return f

    def __createComputeFixedPik(self, inter):
        # xs = [x_0, ..., x_{k-1}]
        # x_i is a concatenation of the state and the action
        # inter = [a, b]
        # returns P[a <= X_k^i <= b] when X_0 = x_0, ..., X_{k-1} = x_{k-1}

        def f(packed_xs):
            xs = self.__unpack(packed_xs)
            mu, sigma = self.__extractMuSigma(xs)
            print("MU, SIGMA", mu, sigma)
            return self.__probInter(mu, sigma, inter)

        return f

    def __getCenter(self, s):
        # s = S_i = [(a_1, b_1), ..., (a_{n+m}, b_{n+m})]
        return [0.5 * (ss[0] + ss[1]) for ss in s]

    def __packSets(self, old_S):
        S = []
        for s in old_S:
            for ss in s:
                S.append(ss)
        return S

    def __startingPointFromSets(self, S):
        # S = [S_0, S_1, ..., S_{k-1}]
        # S_0 is a singleton containing x_0
        # S_i = [S_i^1, ..., S_i^{n+m}]
        # S_i^j = (a, b)
        x = []
        for s in S:
            x += self.__getCenter(s)
        return np.array(x)

    def fit(self, X, Y):

        # X = [X[0], ..., X[N-1]] -> dim n (array even if n == 1)
        # Y = [Y[0], ..., Y[N-1]] -> scalars
        # TODO: optim?

        assert (len(X) == len(Y)), "Wrong sizes X Y"

        self.X = X
        self.Y = Y
        self.N = len(self.Y)

        self.__preCompute()

    def __minimize(self, f, start, bounds, N=1000, iter=10):
        # TODO

        x, y = start, f(start)

        for _ in range(iter):
            for _ in range(N):
                xx = [np.random.uniform(inter[0], inter[1]) for inter in bounds]
                yy = f(xx)
                if yy <= y:
                    x, y = xx, yy
            r = fmin_tnc(f, x, approx_grad=True, bounds=bounds)
            xx, yy = r[0], f(r[0])
            if yy < y:
                x, y = xx, yy

        return x, y

    def computePik(self, S, inter):
        # S = [S_0, S_1, ..., S_{k-1}]
        # S_0 is a singleton containing x_0
        # S_i = [S_i^0, ..., S_i^n]
        # S_i^j = (a, b)

        f = self.__createComputeFixedPik(inter)
        start = self.__startingPointFromSets(S)
        print(start)
        print("FIRST TEST", f(start))
        bounds = self.__packSets(S)
        print("BOUNDS", bounds)
        x, y = self.__minimize(f, start, bounds)
        return f, x, y

    # ---------------------- TODO ---------------------------

#     def predictState(self, bounds, p=0.95):
#
#         print("Proba " + str(p) + " -> " + str(alpha))
#
#         def m(x):
#             return mu(x) - alpha * delta(x)
#
#         def M(x):
#             return mu(x) + alpha * delta(x)
#
#         def minusM(x):
#             return - M(x)
#
#         x0 = np.array([0.5 * (x[0] + x[1]) for x in bounds])
#         xa = fmin_tnc(m, x0, approx_grad=True, bounds=bounds)
#         xb = fmin_tnc(minusM, x0, approx_grad=True, bounds=bounds)
#
#         print(xa, xb)
#
#         return (m(xa[0]), M(xb[0]))
