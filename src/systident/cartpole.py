import math
import main


class Param:

    def __init__(self):
        self.p = {}
        self.p['gravity'] = 9.8
        self.p['masscart'] = 1.0
        self.p['masspole'] = 0.1
        self.p['length'] = 0.5  # actually half the pole's length
        self.p['force_mag'] = 10.0
        self.p['tau'] = 0.02  # seconds between state updates

        self.update()

    def update(self):
        self.p['total_mass'] = self.p['masspole'] + self.p['masscart']
        self.p['polemass_length'] = self.p['masspole'] * self.p['length']


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


class Minimizer:

    def __init__(self, f=f):
        self.f = f
        self.r = []
        self.thetas = []

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

    def computeError(self, param):
        s = 0.0
        for x, u, y in self.r:
            xx = f(x, u, param)
            s += self.delta(xx, y)
        return s


r = main.getReplayBuffer()
conf = Param()
m = Minimizer(f)
m.addReplayBuffer(r)
