import math
from misc.coreGP import CoreGP
import numpy as np
from conf import Conf


class EntropySearch:

    def __init__(self, d=1, bounds=[[0, 1]]):

        self.k = 10
        self.n = 10

        self.m = 11  # for each side
        self.bounds = bounds
        self.dim = d
        self.M = self.m ** d

        self.obs = []

        conf = Conf()
        self.gp = CoreGP(conf)

    def addObs(self, theta, value):
        self.obs.append((theta, value))

    def trainGP(self):
        X = [x[0] for x in self.obs]
        Y = [x[1] for x in self.obs]
        print("Training...")
        self.gp.train(X, Y)
        print("Trained!")

    def sampleThetas(self, confs=[[]], id_variables=None):
        if id_variables is None:
            id_variables = [i for i in range(self.dim)]
        if len(id_variables) == 0:
            return confs
        cur, q = id_variables[0], id_variables[1:]
        new_confs = []
        a, b = self.bounds[cur]
        for i in range(self.m):
            t = i / (self.m - 1)
            x = a * t + b * (1 - t)
            new_confs += [conf + [x] for conf in confs]
        return self.sampleThetas(new_confs, q)

    def computeEntropy(self, P):
        entrop = .0
        for p in P:
            if p > 0:
                entrop -= p * math.log(p)
        return entrop

    def computeP(self, samples):
        P = [0.0 for _ in range(self.M)]
        n = 0.
        for sample in samples:
            P[np.argmin(sample)] += 1.
            n += 1.
        return [p / n for p in P]

    def approximateCurrentDistrib(self):

        mu, sigma = self.gp.MU, self.gp.SIGMA

        samples = [np.random.multivariate_normal(mu, sigma)
                   for _ in range(self.n * self.k)]
        return self.computeP(samples)

    def approximateEntropyGiven(self, i_stheta, stheta):

        mu, sigma = self.gp.predict([stheta], return_cov=True)
        cumulated_entropy = .0

        for _ in range(self.k):

            yi = np.random.multivariate_normal(mu, sigma)
            Mu, Sigma = self.gp.predictGiven(i_stheta, yi)
            Mu = [x[0] for x in Mu]

            samples = [np.random.multivariate_normal(Mu, Sigma)
                       for _ in range(self.n)]
            # samples[n][m ** 2 - 1]

            yi = list(yi)
            samples = [list(s) for s in samples]
            samples = [s[:i_stheta] + yi + s[i_stheta:] for s in samples]

            P = self.computeP(samples)
            cumulated_entropy += self.computeEntropy(P)

        return cumulated_entropy / self.k

    def printResults(self, future_entropy, sthetas):
        for i, c_entropy in enumerate(future_entropy):
            print(sthetas[i], c_entropy / self.k,
                  "[" + str(self.gp.MU[i]) + ' +- '
                  + str(self.gp.SIGMA[i, i]) + "]")

    def findBestEntropy(self):

        sthetas = self.sampleThetas()
        future_entropy = []

        print("Precomputing...")
        self.gp.preComputeGiven(sthetas)
        print("Precomputed!")

        P = self.approximateCurrentDistrib()
        i = np.argmax(P)
        current_entropy = self.computeEntropy(P)
        current_min = sthetas[i]
        current_prob = P[i]

        print("Current entropy", current_entropy)
        print("Current minimum", current_min, current_prob)

        for i_stheta, stheta in enumerate(sthetas):
            if i_stheta % 10 == 0:
                print(str(i_stheta + 1) + "/" + str(len(sthetas)))
            approx_entropy = self.approximateEntropyGiven(i_stheta, stheta)
            future_entropy.append(approx_entropy)

        print(min(future_entropy), np.average(future_entropy),
              max(future_entropy))

        i = np.argmin(future_entropy)
        return sthetas[i], current_min, current_prob, current_entropy

    def iterate(self):
        self.trainGP()
        new_theta, c_min, c_prob, c_entropy = self.findBestEntropy()
        print("NEW THETA", new_theta)
        return new_theta, c_min, c_prob, c_entropy
