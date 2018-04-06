import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
import random as rd

def quantize(x, n_div, mini, maxi):
    y = (x - mini) / (maxi - mini)
    y = min(0.99, max(0, y))
    return int(y * n_div)

class RandomCollecting():
    def new_episode(self, obs):
        return

    def __init__(self):
        self.obs = []

    def act(self, obs):
        return rd.randint(0, 1)

    def update(self, obs, reward, done):
        self.obs.append(obs)

class AgentLearner():

    def new_episode(self, obs):
        print("NEW")
        a_0 = [x[0] for x in self.q]
        a_1 = [x[1] for x in self.q]
        print((sum(a_0) + sum(a_1)) / (len(a_0) + len(a_1)))
        print(min(self.q))
        print(max(self.q))
        #self.q = [[100 for a in range(self.na)] for s in range(self.ns)]
        self.i += 1
        return

    def __init__(self):

        self.i = 0

        self.n_episodes = 100
        self.last_episodes = []
        self.total_reward = 0

        self.n_div = 4
        self.mini = [-0.2, -1, -0.25, -2]
        self.maxi = [0.2, 1, 0.25, 2]

        self.ns = self.n_div ** 4
        self.na = 2

        self.alpha = 0.1
        self.epsilon = 0.1
        self.gamma = 0.99

        self.prev_s = 0
        self.prev_a = 0

        self.q = [[200 for a in range(self.na)] for s in range(self.ns)]

    def discretize(self, obs):
        #print(obs)
        s = 0
        m = 1
        for i in range(4):
            x = obs[i]
            s += m * quantize(x, self.n_div, self.mini[i], self.maxi[i])
            m *= self.n_div
        #print(s)
        return s

    def act(self, obs):
        s = self.discretize(obs)
        a = None
        if rd.random() < self.epsilon:
            a = rd.randint(0, 1)
        else:
            if self.q[s][0] > self.q[s][1]:
                a = 0
            else:
                a = 1
        self.prev_a = a
        self.prev_s = s
        return a

    def update(self, obs, reward, done):
        self.total_reward += reward
        s = self.discretize(obs)
        if done:
            self.q[self.prev_s][self.prev_a] = (1 - self.alpha) * self.q[self.prev_s][self.prev_a] + self.alpha * reward
            self.end()
        else:
            self.q[self.prev_s][self.prev_a] = (1 - self.alpha) * self.q[self.prev_s][self.prev_a] + self.alpha * (reward + self.gamma * max(self.q[s]))

    def end(self):
        self.last_episodes.append(self.total_reward)
        self.last_episodes = self.last_episodes[len(self.last_episodes) - self.n_episodes:]
        self.total_reward = 0
        print("Current average", sum(self.last_episodes) / len(self.last_episodes))

class Agent():
    def __init__(self):

        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = (self.masspole + self.masscart)
        self.length = 0.5 # actually half the pole's length
        self.polemass_length = (self.masspole * self.length)
        self.force_mag = 10.0
        self.tau = 0.02  # seconds between state updates

        self.x = None
        self.x_dot = None
        self.theta = None
        self.theta_dot = None

    def distance_to_center(self, x, theta):
        xx = x.item(0) / 2.4
        yy = theta / 0.2
        return xx * xx + yy * yy

    def simulate(self, action):
        force = self.force_mag if action==1 else -self.force_mag
        costheta = math.cos(self.theta)
        sintheta = math.sin(self.theta)
        temp = (force + self.polemass_length * self.theta_dot * self.theta_dot * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta* temp) / (self.length * (4.0/3.0 - self.masspole * costheta * costheta / self.total_mass))
        xacc  = temp - self.polemass_length * thetaacc * costheta / self.total_mass
        x  = self.x + self.tau * self.x_dot
        x_dot = self.x_dot + self.tau * xacc
        theta = self.theta + self.tau * self.theta_dot
        theta_dot = self.theta_dot + self.tau * thetaacc
        return (x, x_dot, theta, theta_dot)

    def act(self, obs):

        (x_0, x_dot_0, theta_0, theta_dot_0) = self.simulate(0)
        (x_1, x_dot_1, theta_1, theta_dot_1) = self.simulate(1)

        x0 = x_0 + self.tau * x_dot_0
        theta0 = theta_0 + self.tau * theta_dot_0

        x1 = x_1 + self.tau * x_dot_1
        theta1 = theta_1 + self.tau * theta_dot_1

        d0 = self.distance_to_center(x0, theta0)
        d1 = self.distance_to_center(x1, theta1)

        if d0 < d1:
            return 0
        else:
            return 1

    def update(self, obs, reward, done):
        self.x, self.x_dot, self.theta, self.theta_dot = obs

    def new_episode(self, obs):
        self.x, self.x_dot, self.theta, self.theta_dot = obs

class LinearAgent():

    def __init__(self):

        gravity = 9.8
        masscart = 1.0
        masspole = 0.1
        total_mass = (masspole + masscart) # = 1.1
        length = 0.5 # actually half the pole's length
        force_mag = 10.0
        tau = 0.02  # seconds between state updates

        A = (masspole * length) / total_mass
        B = tau / (length * 4.0/3.0 - A) # = 1.5

        temp = force_mag / total_mass # = 9

        a_x = tau * temp + A * B  * temp
        theta_x = - A * B * gravity

        theta_theta = B * gravity
        a_theta =  - B * temp

        self.A = np.matrix([
            [1, tau, 0, 0],
            [0, 1, theta_x, 0],
            [0, 0, 1, tau],
            [0, 0, theta_theta, 1]
        ])
        self.B = np.matrix([0, a_x, 0, a_theta]).T

        print("A", self.A)
        print("B", self.B)

        self.X = np.zeros([4, 1])
        self.measured_X = np.zeros([4, 1])

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

        return a

    def rectify(self):

        print("COMPARISON")
        a = self.X.item(1)
        b = self.measured_X.item(1)
        print(abs(a  - b))
        a = self.X.item(3)
        b = self.measured_X.item(3)
        print(abs(a - b))

    def update(self, obs, reward, done):
        self.measured_X = np.matrix(obs).T
        self.rectify()

    def new_episode(self, obs):
        self.measured_X = np.matrix(obs).T

class LearningLinearAgent():
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
            p = np.linalg.lstsq(self.x.T, self.y.T)[0]
            self.A = p[:-1].T
            self.B = p[-1].T
            print(self.A)
            print(self.B)
        self.measured_X = Y

    def new_episode(self, obs):
        self.measured_X = np.matrix(obs).T
        self.X = None

env = gym.make('CartPole-v0')
a = AgentLearner()
#a = LinearAgent()
#a = LearningLinearAgent()
#a = Agent()
#a = RandomCollecting()

for i_episode in range(100000):
    observation = env.reset()
    a.new_episode(observation)
    t = 0
    while True:
        t += 1
        env.render()
        action = a.act(observation)
        observation, reward, done, info = env.step(action)
        a.update(observation, reward, done)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
