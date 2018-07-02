import numpy as np
import random as rd
from misc import agent, tools


def default_convert_control(u):
    return u.item(0)


def default_convert_action(a):
    return [a]


def default_convert_obs(obs):
    return np.matrix(obs).T


def default_random_action():
    return 2 * rd.random() - 1


class LinearAgent(agent.Agent):

    def __init__(self,
                 n=4,
                 epsilon=0,
                 Q=None,
                 N=10,
                 convert_action=default_convert_action,
                 convert_control=default_convert_control,
                 convert_obs=default_convert_obs,
                 random_action=default_random_action,
                 ):

        self.n = n

        self.x = np.zeros([self.n + 1, 0])
        self.y = np.zeros([self.n, 0])

        self.A = np.identity(self.n)
        self.B = np.zeros([self.n, 1])

        self.measured_X = None
        self.last_action = None

        self.convert_action = convert_action
        self.convert_control = convert_control
        self.convert_obs = convert_obs

        self.epsilon = epsilon
        self.random_action = random_action

        self.N = N
        if Q is None:
            self.Q = np.identity(n)
        else:
            self.Q = Q

    def act(self, obs):

        z = 0.0000000000000000000000000000000000000000000000001
        eps = np.random.rand(self.n, self.n) * z
        eps_2 = np.random.rand(self.n, self.n) * z
        eps_col = np.random.rand(self.n, 1) * z
        eps_col_2 = np.random.rand(self.n, 1) * z

        c = tools.LQR(self.A + eps, self.B + eps_col, self.Q + eps_2, self.N)
        x = self.convert_obs(obs) + eps_col_2
        u = c.solve(x)

        a = self.convert_control(u)

        if rd.random() < self.epsilon:
            a = self.random_action()

        self.last_action = a

        return self.convert_action(a)

    def update(self, obs, reward, done):
        Y = self.convert_obs(obs)
        if not (self.measured_X is None):
            X = self.measured_X
            X = np.vstack([X, self.last_action])
            self.x = np.hstack([self.x, X])
            self.y = np.hstack([self.y, Y])
            r = np.linalg.lstsq(self.x.T, self.y.T, rcond=None)
            # print(r[1:])
            p = r[0]
            (self.A, self.B) = (p[:-1].T, p[-1].T)
        self.measured_X = Y

    def new_episode(self, obs):
        self.measured_X = None
