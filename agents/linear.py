import numpy as np
import agent
import math
import tools
import random as rd

class LearningLinearAgent(agent.Agent):

    def __init__(self):

        self.theta_threshold = 12 * 2 * math.pi / 360
        self.x_threshold = 2.4

        n_div = 6

        #mini = [-0.2, -1, -0.25, -2]
        #maxi = [0.2, 1, 0.25, 2]

        x_max = tools.getMax(self.x_threshold, n_div)
        theta_max = tools.getMax(self.theta_threshold, n_div)

        mini = [-x_max, -1, -theta_max, -2]
        maxi = [x_max, 1, theta_max, 2]

        self.quantizer = tools.Quantizer(n_div, mini, maxi)

        self.ns = n_div ** 4
        self.actions = [0, 1]

        self.P = []
        self.P.append(np.identity(self.ns))
        self.P.append(np.identity(self.ns))

        self.p_done = np.zeros([self.ns, 1])
        n_done = 100.
        incr_done = 1. / n_done
        for s in range(self.ns):
            for _ in range(int(n_done)):
                x = self.quantizer.undiscretizeRandom(s)
                if self.done(x):
                    self.p_done[s] += incr_done

        self.gamma = 0.99
        self.q = [[1. / (1. - self.gamma) for _ in self.actions] for x in range(self.ns)]

        self.n = 4
        self.x = np.zeros([self.n + 1, 0])
        self.y = np.zeros([self.n, 0])

        self.A = np.identity(self.n)
        self.B = np.zeros([self.n, 1])

        self.X = None
        self.measured_X = None
        self.last_action = None

        self.n_approx = 20
        self.H = 20

    def done(self, state):
        x = state[0]
        theta = state[2]
        return x < -self.x_threshold \
                or x > self.x_threshold \
                or theta < -self.theta_threshold \
				or theta > self.theta_threshold

    def getValue(self, x):
        done = self.done(x)
        s = self.quantizer.discretize(x)
        if done:
            return 1
        else:
            return 1 + self.gamma * max(self.q[s])

    def getValueState(self, s):
        return 1 + (1 - self.p_done[s]) * self.gamma * max(self.q[s])

    def updateTransitionMatrix(self, a):
        self.P[a] = 0.5 * self.P[a]
        incr = 0.5 / self.n_approx
        for s in range(self.ns):
            for _ in range(self.n_approx):
                x = np.matrix(self.quantizer.undiscretizeRandom(s)).T
                s_succ = self.quantizer.discretize(self.simulate(x, a))
                self.P[a][s][s_succ] += incr

    def learn(self):
        for a in self.actions:
            self.updateTransitionMatrix(a)
        epsilon = 1.
        delta = epsilon
        i = 0
        while delta >= epsilon:
            i += 1
            print("Iteration", i, "delta =", delta, ", max =", max(self.q), ", min =",
                  min(self.q))
            delta = 0
            indices = np.arange(self.ns)
            np.random.shuffle(indices)
            l = []
            for s in indices:
                for a in self.actions:
                    v = 0.0
                    for ss in range(self.ns):
                        pss = self.P[a][s][ss]
                        if pss > 0:
                            v += pss * self.getValueState(ss)
                    old_v = self.q[s][a]
                    self.q[s][a] = self.p_done[s] +  (1 - self.p_done[s]) * v
                    delta = max(delta, abs(old_v - self.q[s][a]))

    #def distance_to_center(self, X):
    #    xx = X.item(0) / 2.4
    #    yy = X.item(2) / 0.2
    #    return xx * xx + yy * yy

    def simulate(self, X, action):
        a = 2 * action - 1
        return self.A * X + a * self.B

    def oldEvaluate(self, state, i):
        if i == 0:
            return not self.done(state)
        x0 = self.simulate(state, 0)
        x1 = self.simulate(state, 1)
        c0 = (not self.done(x0)) and self.evaluate(x0, i - 1)
        c = c0 or ((not self.done(x1)) and self.evaluate(x1, i - 1))
        return c

    def goodEvaluate(self, state, a, i):
        if i == 0:
            return not self.H
        x = self.simulate(state, a)
        if self.done(x):
            return self.H - i
        return self.evaluate(x, a, i - 1)

    def evaluate(self, x, i):
        if i == 0:
            return self.H
        if self.done(x):
            return self.H - i
        x0 = self.simulate(x, 0)
        x1 = self.simulate(x, 1)
        v0 = self.evaluate(x0, i - 1)
        if v0 == self.H:
            return v0
        return max(v0, self.evaluate(x1, i - 1))

    def actSimulate(self, obs):

        x0 = self.simulate(self.measured_X, 0)
        x1 = self.simulate(self.measured_X, 1)

        v0 = self.evaluate(x0, self.H)
        v1 = self.evaluate(x1, self.H)

        #print(v0, v1)

        a = 0
        self.X = self.simulate(self.measured_X, 0)
        if v1 > v0:
            a = 1
        elif v1 == v0:
            a = rd.randint(0, 1)

        if a == 1:
            self.X = self.simulate(self.measured_X, 0)

        self.last_action = 2 * a - 1

        return a

    def act(self, obs):
        x = self.convert_obs(obs)
        s = self.quantizer.discretize(x)
        v0 = self.q[s][0]
        v1 = self.q[s][1]
        a = 0
        if v1 > v0:
            a = 1
        self.last_action = 2 * a - 1
        self.X = self.simulate(x, a)
        return a

    def oldAct(self, obs):

        x0 = self.simulate(self.measured_X, 0)
        s0 = self.quantizer.discretize(x0)
        v0 = self.values[s0]

        x1 = self.simulate(self.measured_X, 1)
        s1 = self.quantizer.discretize(x1)
        v1 = self.values[s1]

        a = 0
        self.X = x0
        #print(s0, s1)
        #print(self.done(x0), self.done(x1))
        #print(v0, v1)
        if v1 > v0:
            a = 1
            self.X = x1

        self.last_action = 2 * a - 1

        return a

    def convert_obs(self, obs):
        return np.matrix(obs).T

    def update(self, obs, reward, done):
        Y = self.convert_obs(obs)
        if not (self.X is None):
            X = self.measured_X
            X = np.vstack([X, self.last_action])
            self.x = np.hstack([self.x, X])
            self.y = np.hstack([self.y, Y])
            p = np.linalg.lstsq(self.x.T, self.y.T, rcond=None)[0]
            self.A = p[:-1].T
            self.B = p[-1].T
        self.measured_X = Y

    def new_episode(self, obs):
        self.measured_X = np.matrix(obs).T
        self.X = None

    def end_episode(self):
        self.learn()
