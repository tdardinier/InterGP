import math
import main
from misc.coreGP import CoreGP
import numpy as np
from conf import Conf


variables = ['masscart', 'length']
bounds = [(0.5, 2), (0.1, 1.0)]


class Param:

    @staticmethod
    def fromVector(r):
        conf = Param()
        i = 0
        for x in variables:
            conf.p[x] = r[i]
            # print(x, '->',  r[i])
            i += 1
        conf.update()
        return conf

    def __init__(self):
        self.p = {}
        self.default()
        self.update()

    def default(self):
        self.p['gravity'] = 9.8
        self.p['masscart'] = 1.0
        self.p['masspole'] = 0.1
        self.p['length'] = 0.5  # actually half the pole's length
        self.p['force_mag'] = 10.0
        self.p['tau'] = 0.02  # seconds between state updates

    def update(self):
        self.p['total_mass'] = self.p['masspole'] + self.p['masscart']
        self.p['polemass_length'] = self.p['masspole'] * self.p['length']

    def toVector(self):
        r = []
        for x in variables:
            r.append(self.p[x])
        return r


def f(state, action, param):

    x, x_dot, theta, theta_dot = state

    costheta = math.cos(theta)
    sintheta = math.sin(theta)

    force = param.p['force_mag'] if action == 1 else -param.p['force_mag']
    costheta = math.cos(theta)
    sintheta = math.sin(theta)
    temp = (force + param.p['polemass_length'] * theta_dot * theta_dot * sintheta) / param.p['total_mass']
    thetaacc = (param.p['gravity'] * sintheta - costheta * temp) / (param.p['length'] * (4.0 / 3.0 - param.p['masspole'] * costheta * costheta / param.p['total_mass']))
    xacc = temp - param.p['polemass_length'] * thetaacc * costheta / param.p['total_mass']
    x = x + param.p['tau'] * x_dot
    x_dot = x_dot + param.p['tau'] * xacc
    theta = theta + param.p['tau'] * theta_dot
    theta_dot = theta_dot + param.p['tau'] * thetaacc
    new_state = (x, x_dot, theta, theta_dot)

    return new_state


initial_k = 20


class Minimizer:

    def __init__(self, f=f):

        self.n = 50
        self.m = 3  # for each side
        self.M = self.m ** len(variables)

        self.f = f
        self.r = []
        self.thetas = []

        self.bounds = bounds

        conf = Conf()
        self.gp = CoreGP(conf)

    def addTheta(self, conf):
        self.thetas.append(conf.toVector())

    def addReplayBuffer(self, r, c=200):
        x = [xx.tolist() for xx in r.x[:c]]
        x = [[xxx[0] for xxx in xx] for xx in x]
        u = [uu.item(0) for uu in r.u[:c]]
        y = [yy.tolist() for yy in r.y[:c]]
        y = [[yyy[0] for yyy in yy] for yy in y]
        self.r = list(zip(x, u, y))

    def delta(self, x1, x2):
        s = 0.0
        for xx1, xx2 in zip(x1, x2):
            s += (xx2 - xx1) ** 2
        return math.sqrt(s)

    def computeError(self, r):
        conf = Param.fromVector(r)
        s = 0.0
        for x, u, y in self.r:
            xx = f(x, u, conf)
            s += self.delta(xx, y)
        return s

    def trainGP(self):
        Y = [self.computeError(r) for r in self.thetas]
        # print(self.thetas, Y)
        print('Y', Y)
        self.gp.train(self.thetas, Y)

    def sampleThetas(self, confs=[[]], id_variables=[i for i in range(len(variables))]):
        if len(id_variables) == 0:
            return confs
        cur, q = id_variables[0], id_variables[1:]
        new_confs = []
        a, b = bounds[cur]
        # print(a, b)
        for i in range(self.m):
            t = i / (self.m - 1)
            # print('t', t)
            x = a * t + b * (1 - t)
            # print(x)
            new_confs += [conf + [x] for conf in confs]
        return self.sampleThetas(new_confs, q)

    def getDistrib(self, sthetas):
        Mu, Sigma = self.gp.predict(sthetas, return_cov=True)
        print(Sigma[4])
        r = [(Mu[i], Sigma[i, i]) for i in range(len(sthetas))]
        for i in range(len(sthetas)):
            theta = sthetas[i]
            print(theta, self.computeError(theta), r[i])
        return Mu, Sigma

    def sampleErrorThetas(self):

        sthetas = self.sampleThetas()
        Mu, Sigma = self.getDistrib(sthetas)

        samples = [np.random.multivariate_normal(Mu, Sigma)
                   for _ in range(self.n)]
        # samples[n][m ** 2]

        P = [0.0 for _ in range(self.M)]
        for sample in samples:
            P[np.argmin(sample)] += 1.
        P = [p / self.n for p in P]
        print(P)

        def entrop(p):
            if p > 0:
                return p * math.log(p)
            else:
                return 0

        entropic_terms = [entrop(p) for p in P]
        return sthetas[np.argmin(entropic_terms)]

    def iterate(self):
        self.trainGP()
        new_theta = self.sampleErrorThetas()
        print("NEW THETA", new_theta)
        self.thetas.append(new_theta)


r = main.getReplayBuffer()
m = Minimizer(f)
m.addReplayBuffer(r)

for _ in range(initial_k):
    conf = Param()
    for i in range(len(variables)):
        var = variables[i]
        a, b = bounds[i]
        conf.p[var] = np.random.uniform(a, b)
    m.addTheta(conf)
