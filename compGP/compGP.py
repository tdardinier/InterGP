from compGP.gp import GP
from copy import deepcopy


class CompGP:

    def __init__(self, k, n, m=1, debug=False, scipy=False):

        self.n = n
        self.m = m

        self.GPs = [GP(k, i, n, m, debug=debug, scipy=scipy) for i in range(n)]

    def fit(self, X, U, Y):
        # X = [X_0, ..., X_{N-1}]
        # U = [U_0, ..., U_{N-1}]
        # Y = [Y_0, ..., Y_{N-1}]
        # {X, U, Y}_i = np.matrix([X_i^0, ..., X_i^{n-1}]).T

        XU = [x + u for (x, u) in zip(X, U)]
        for i in range(self.n):
            YY = [y[i] for y in Y]
            self.GPs[i].fit(XU, YY)

    def __combineSetsStatesActions(self, original_S, U):

        S = deepcopy(original_S)

        SS = []
        for s, u in zip(S, U):
            # s = [(a1, b1), ..., (an, bn)]
            # u = [u_1, ..., u_m]
            ss = s
            for i in range(self.m):
                uu = u[i]
                ss.append((uu, uu))
            SS.append(ss)

        return SS

    def synthesizeNextSet(self, S, U, p=0.95):
        # S = [S_0, ..., S_{k-1}]
        # S_0 is a singleton containing x_0
        # S_i = [S_i^0, ..., S_i^n]
        # S_i^j = (a, b)

        # U = [U_0, ..., U_{k-1}]

        pp = p ** (1. / self.n)
        SS = self.__combineSetsStatesActions(S, U)
        next_S = []

        for i in range(self.n):
            inter = self.GPs[i].synthesizeSet(SS, pp)
            print("Generated:", inter)
            next_S.append(inter)

        return next_S

    def computeNextProb(self, S, U, S_k):
        # S = [S_0, ..., S_{k-1}]
        # S_0 is a singleton containing x_0
        # S_i = [S_i^0, ..., S_i^n]
        # S_i^j = (a, b)

        # U = [U_0, ..., U_{k-1}]

        SS = self.__combineSetsStatesActions(S, U)
        p = 1

        for i in range(self.n):
            pp = self.GPs[i].computePik(SS, S_k[i])[2]
            print(str(i) + " -> " + str(pp))
            p *= pp

        return p

    def synthesizeSets(self, x_0, U, k, p):
        # x_0 = [x_0^1, ..., x_0^n]

        SS = [[(xx, xx) for xx in x_0]]
        UU = []
        probs = []

        cum_p = 1.

        for i in range(k):
            if i == k - 1:
                pp = p / cum_p
            else:
                pp = (p / cum_p) ** (.7)
            # pp = (p / cum_p) ** (1. / (k - i))
            print("-" * 80)
            print("Step", i+1)
            UU.append(U[i])
            s = self.synthesizeNextSet(SS, UU, pp)
            probs.append(self.computeNextProb(SS, UU, s))
            cum_p *= probs[-1]
            print("Aim:", pp, ", real:", probs[-1], ", cumulative", cum_p)
            SS.append(s)

        print("Total prob", cum_p, ", ", probs)

        return SS[1:], probs
