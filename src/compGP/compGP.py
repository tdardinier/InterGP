from compGP.gp import GP
from compGP.trajectory import Trajectory
from copy import deepcopy


class CompGP:

    def __init__(self, conf):

        self.n = conf.n
        self.m = conf.m

        self.GPs = [GP(conf, i) for i in range(conf.n)]

        self.riskAllocUniform = conf.riskAllocUniform
        self.probTransition = conf.probTransition

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
        p = []

        for i in range(self.n):
            pp = self.GPs[i].computePik(SS, S_k[i])[2]
            print(str(i) + " -> " + str(pp))
            p.append(pp)

        return p

    def computeProbTraj(self, S, U):
        p = 1.
        P, SS, UU = [], [], []

        for i in range(len(S) - 1):
            UU.append(U[i])
            SS.append(S[i])
            P.append(self.computeNextProb(SS, UU, S[i+1]))
            p *= P[-1]

        return p, P

    def synthesizeSets(self, x_0, U, k, p):
        # x_0 = [x_0^1, ..., x_0^n]

        traj = Trajectory()
        traj.addStart(x_0)
        traj.addU(U[:k])

        SS = [[(xx, xx) for xx in x_0]]
        UU = []

        cum_p = 1.

        for i in range(k):
            if self.probTransition:
                pp = p
            else:
                cum_p = max(cum_p, p)  # to avoid div by 0
                if self.riskAllocUniform:
                    pp = (p / cum_p) ** (1. / (k - i))
                else:
                    if i == k - 1:
                        pp = p / cum_p
                    else:
                        pp = (p / cum_p) ** (.5)
                pp = min(pp, 0.99999)
            print("-" * 80)
            print("Step", i+1)
            UU.append(U[i])
            s = self.synthesizeNextSet(SS, UU, pp)

            aimedP = [pp ** (1. / self.n) for _ in range(self.n)]
            realP = self.computeNextProb(SS, UU, s)
            traj.addPrediction(s, aimedP, realP)

            prob = 1
            for ppp in realP:
                prob *= ppp
            cum_p *= prob

            print("Aim:", pp, ", real:", prob, ", cumulative", cum_p)
            SS.append(s)

        print()
        print("Total prob", cum_p)

        return traj
