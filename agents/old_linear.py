import numpy as np
import agent

class LearningLinearAgent(agent.Agent):
    def __init__(self):

        self.n = 4
        self.x = np.zeros([self.n + 1, 0])
        self.y = np.zeros([self.n, 0])

        self.A = np.identity(self.n)
        self.B = np.zeros([self.n, 1])

        self.X = None
        self.measured_X = None
        self.last_action = None

    def distance_to_center(self, X):
        xx = X.item(0) / 2.4
        yy = X.item(2) / 0.2
        return xx * xx + yy * yy

    def simulate(self, X, action):
        a = 2 * action - 1
        return self.A * X + a * self.B

    def act(self, obs):

        X_0 = self.simulate(self.measured_X, 0)
        XX_0 = self.simulate(X_0, 0)
        d0 = self.distance_to_center(XX_0)

        X_1 = self.simulate(self.measured_X, 1)
        XX_1 = self.simulate(X_1, 1)
        d1 = self.distance_to_center(XX_1)

        a = 0
        self.X = X_0
        if d0 > d1:
            a = 1
            self.X = X_1

        self.last_action = 2 * a - 1

        return a

    def update(self, obs, reward, done):
        Y = np.matrix(obs).T
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
