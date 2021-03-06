import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
from math import sqrt
from misc.coreGP import CoreGP


class GP:

    def __noisifyMatrix(self, M):
        np.random.seed(self.conf.seed)
        n, m = M.shape
        r = np.matrix([[np.random.rand() for _ in range(m)] for _ in range(n)])
        r *= self.conf.epsilon
        return M + r

    def __normalizeSigma(self, sigma):
        np.random.seed(self.conf.seed)
        if sigma < 0.1 * self.conf.epsilon:
            if self.conf.debug:
                print("ALERT: SMALL SIGMA", sigma)
            return (np.random.rand() + 0.1) * self.conf.epsilon
        return sigma

    def __init__(self, conf, i):

        self.i = i  # ith component

        self.X = None
        self.Y = None
        self.N = None

        self.gp = CoreGP(conf)

        self.conf = conf

    def fit(self, X, Y):
        self.gp.train(X, Y)

    def computePik(self, S, inter):
        # S = [S_0, S_1, ..., S_{k-1}]
        # S_0 is a singleton containing x_0
        # S_i = [S_i^0, ..., S_i^n]
        # S_i^j = (a, b)

        # self.__description(S)

        f, approx_f = self.__createComputeFixedPik(inter)
        start = self.__startingPointFromSets(S)
        bounds = self.__packSets(S)

        # xx, yy = self.__minimize(approx_f, start, bounds)
        x, y = self.__minimize(f, start, bounds)

        if self.conf.debug:
            print("bounds", bounds)
            print("inter", inter)
            print("bounds", bounds)
            print("REAL F", x, f(x), y)
            # print("APPROX F", xx, f(xx), yy)

        # yy = f(xx)
        y = f(x)
        # if yy < y:
        #    x, y = xx, yy
        return f, x, y

    def synthesizeSet(self, S, p=0.95):
        # S = [S_0, S_1, ..., S_{k-1}]
        # S_0 is a singleton containing x_0
        # S_i = [S_i^0, ..., S_i^n]
        # S_i^j = (a, b)

        # self.__description(S)

        start = self.__startingPointFromSets(S)
        bounds = self.__packSets(S)
        if self.conf.debug:
            print("synthe: bounds", bounds)

        if self.conf.centered or p < 0.5:
            m = self.__createM(p, bigM=False)
            M = self.__createM(p, bigM=True)
        else:
            f_a_tilde = self.__createTilde(p, a=True)
            f_b_tilde = self.__createTilde(p, a=False)
            a_tilde = self.__minimize(f_a_tilde, start, bounds)[1]
            b_tilde = self.__maximize(f_b_tilde, start, bounds)[1]
            # print("TILDE", a_tilde, b_tilde)
            m = self.__createMInterval(a_tilde, b_tilde, p, bigM=False)
            M = self.__createMInterval(a_tilde, b_tilde, p, bigM=True)

        a = self.__minimize(m, start, bounds)
        b = self.__maximize(M, start, bounds)

        # self.debug = True
        # if self.debug:
        # print("MIN:", m(a[0]))
        # self.debug = False

        # print("Resulting set: ", [a, b])

        return [a[1], b[1]]

    def __probInter(self, mu=0, sigma=1, inter=[-1.96, 1.96]):
        if self.conf.debug:
            print("prob", mu, sigma, inter)
        ssigma = sqrt(sigma)
        return norm.cdf(inter[1], mu, ssigma) - norm.cdf(inter[0], mu, ssigma)

    def __extractMuSigma(self, xs):
        # xs = [x_0, ..., x_{k-1}]
        # x_i = np.matrix([[], ..., []])
        # x_i is a concatenation of state and action

        k = len(xs)

        MU, SIGMA = self.gp.predict(xs, return_cov=True)

        if self.conf.debug:
            print("BIG MU SIGMA", MU, SIGMA)
            print("k", k)

        if k == 1:
            return MU.item(0), SIGMA.item(0)

        MU_1 = MU[np.ix_(range(k-1))].reshape([k-1, 1])
        MU_2 = MU[np.ix_([k-1])]

        SIGMA_11 = SIGMA[np.ix_(range(k-1), range(k-1))]
        SIGMA_12 = SIGMA[np.ix_(range(k-1), [k-1])]
        SIGMA_21 = SIGMA[np.ix_([k-1], range(k-1))]
        SIGMA_22 = SIGMA[np.ix_([k-1], [k-1])]

        prod = SIGMA_21 * np.linalg.inv(self.__noisifyMatrix(SIGMA_11))
        x = np.matrix([xx[self.i] for xx in xs[1:]]).T

        if self.conf.debug and False:
            print("\nMU 2\n", MU_2)
            print("\nSIGMA_21\n", SIGMA_21)
            print("\nSIGMA_11\n", SIGMA_11)
            print("\nPROD\n", prod)
            print("\nMU_1\n", MU_1)
            print("\nx - MU_1\n", x - MU_1)
            print("\nprod * (x - MU_1)\n", prod * (x - MU_1))
            print("x", x)

        mu = MU_2 + prod * (x - MU_1)
        sigma = SIGMA_22 - prod * SIGMA_12

        if self.conf.debug:
            print("small mu sigma", mu, sigma)

        # TODO detect epsilon
        # a = SIGMA_22.item(0)
        # b = sigma.item(0)
        # if b > 10 * a:
            # print()
            # print("BIG SIGMA")
            # print(a, '->', b)

        # if b > 0.1:
            # print("BIG VARIANCE", b)

        if self.conf.test_chaos_theory and k >= self.conf.k_begin_chaos:
            return mu.item(0), self.conf.epsilon

        return mu.item(0), self.__normalizeSigma(sigma.item(0))

    def __unpack(self, xx):
        length = self.conf.n + self.conf.m
        k = len(xx) // length
        x = [[xx[i * length + j] for j in range(length)] for i in range(k)]
        if self.conf.debug:
            print("UNPACKED", x)
        return x

    def __getAlpha(self, p=0.95):
        return norm.ppf(0.5 * (1. + p))

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

    def __sampleExtremities(self, bounds, n_iter):
        np.random.seed(self.conf.seed)

        if n_iter == 0:
            return []

        tail = self.__sampleExtremities(bounds, n_iter - 1)

        x = []
        for (a, b) in bounds:
            if np.random.rand() < 0.5:
                x.append(a)
            else:
                x.append(b)

        tail.append(x)

        return tail

    def __minimize(self, f, start, bounds):
        np.random.seed(self.conf.seed)

        x, y = start, f(start)

        def compare(x, y, xx):
            yy = f(xx)
            if yy <= y:
                return xx, yy
            return x, y

        # assert (abs(f(x) - y) < self.conf.epsilon_f)

        extremities = self.__sampleExtremities(bounds,
                                               self.conf.max_iter_minimizer)

        for xx in extremities:
            x, y = compare(x, y, xx)

        # assert (abs(f(x) - y) <= self.conf.epsilon_f)

        for _ in range(self.conf.max_iter_minimizer):
            xx = [np.random.uniform(inter[0], inter[1]) for inter in bounds]
            x, y = compare(x, y, xx)

        # assert (abs(f(x) - y) < self.conf.epsilon_f)

        r = minimize(f, x, bounds=bounds,
                     options={'maxiter': self.conf.max_iter_minimizer})
        x, y = compare(x, y, r['x'])

        # assert (abs(f(x) - y) < self.conf.epsilon_f)

        return x, y

    def __maximize(self, f, start, bounds):

        def new_f(x):
            return - f(x)

        x, y = self.__minimize(new_f, start, bounds)
        return x, -y

    def __description(self, S):
        print("This sequence of sets:", S)
        y = [s[self.i] for s in S]
        for i in range(len(S) - 1):
            print(str(S[i]) + " -> " + str(y[i+1]))

    def __findInterval(self, a_tilde, b_tilde, p, mu=0, sigma=1):

        assert (b_tilde >= a_tilde)

        ssigma = sqrt(sigma)

        def cdf(x):
            return norm.cdf(x, mu, ssigma)

        def ppf(pp):
            return norm.ppf(pp, mu, ssigma)

        p_tilde = cdf(b_tilde) - cdf(a_tilde)
        if p_tilde >= p:
            return [a_tilde, b_tilde]

        da = abs(mu - a_tilde)
        db = abs(b_tilde - mu)
        delta = max(da, db)
        if cdf(mu + delta) - cdf(mu - delta) >= p:
            if da < db:
                return [ppf(cdf(b_tilde) - p), b_tilde]
            return [a_tilde, ppf(p + cdf(a_tilde))]

        return [ppf(0.5 * (1 - p)), ppf(0.5 * (1 + p))]

    def __createMInterval(self, a_tilde, b_tilde, p, bigM=True):

        if bigM:
            def f(packed_xs):
                xs = self.__unpack(packed_xs)
                mu, sigma = self.__extractMuSigma(xs)
                (a, b) = self.__findInterval(
                    a_tilde, b_tilde, p, mu, sigma)
                # print("INTER", p, self.__probInter(mu, sigma, [a, b]))
                return b

        else:
            def f(packed_xs):
                xs = self.__unpack(packed_xs)
                mu, sigma = self.__extractMuSigma(xs)
                (a, b) = self.__findInterval(
                    a_tilde, b_tilde, p, mu, sigma)
                # print("INTER", p, self.__probInter(mu, sigma, [a, b]))
                return a

        return f

    def __createTilde(self, p, a=True):

        if a:
            def f(packed_xs):
                xs = self.__unpack(packed_xs)
                mu, sigma = self.__extractMuSigma(xs)
                return norm.ppf(1-p, mu, sqrt(sigma))

        else:
            def f(packed_xs):
                xs = self.__unpack(packed_xs)
                mu, sigma = self.__extractMuSigma(xs)
                return norm.ppf(p, mu, sqrt(sigma))

        return f

    def __createM(self, p=0.95, bigM=True):
        alpha = [self.__getAlpha(p)]  # array so nonlocal
        # print("ALPHA", alpha[0])

        if bigM:
            def f(packed_xs):
                # xs = [x_0, ..., x_{k-1}]
                # x_i is a concatenation of the state and the action
                xs = self.__unpack(packed_xs)
                mu, sigma = self.__extractMuSigma(xs)
                return mu + alpha[0] * sqrt(sigma)
        else:
            def f(packed_xs):
                # xs = [x_0, ..., x_{k-1}]
                # x_i is a concatenation of the state and the action
                xs = self.__unpack(packed_xs)
                mu, sigma = self.__extractMuSigma(xs)
                return mu - alpha[0] * sqrt(sigma)

        return f

    def __createComputeFixedPik(self, inter):
        # xs = [x_0, ..., x_{k-1}]
        # x_i is a concatenation of the state and the action
        # inter = [a, b]
        # returns P[a <= X_k^i <= b] when X_0 = x_0, ..., X_{k-1} = x_{k-1}

        def f(packed_xs):
            xs = self.__unpack(packed_xs)
            mu, sigma = self.__extractMuSigma(xs)
            if self.conf.debug:
                print("MU, SIGMA", mu, sigma)
            p = self.__probInter(mu, sigma, inter)
            return p

        def approx_f(packed_xs):
            xs = self.__unpack(packed_xs)
            mu, sigma = self.__extractMuSigma(xs)
            if self.conf.debug:
                print("MU, SIGMA", mu, sigma)
            p = self.__probInter(mu, sigma, inter)
            p -= self.conf.epsilon * ((inter[0] - mu) / sigma) ** 2
            p -= self.conf.epsilon * ((inter[1] - mu) / sigma) ** 2
            return p

        return f, approx_f
