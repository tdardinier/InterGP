import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
import random as rd

class Stats():
    def __init__(self):
        self.stats = []

    def addEpisode(self, total_reward):
        self.stats.append(total_reward)

    def print(self, n = 100):
        l = self.stats[len(self.stats) - n:]
        print("Current average:", sum(l) / len(l))

class Agent():
    def __init__(self):
        pass

    def new_episode(self, obs):
        pass

    def act(self, obs):
        return rd.randint(0, 1)

    def update(self, obs, reward, done):
        pass

class Controller():
    def __init__(self, env, agent):
        self.env = env
        self.agent = agent
        self.stats = Stats()

    def run_episode(self, render = True):
        observation = env.reset()
        self.agent.new_episode(observation)
        t = 0
        done = False
        total_reward = 0
        while not done:
            t += 1
            if render:
                env.render()
            action = self.agent.act(observation)
            observation, reward, done, info = env.step(action)
            total_reward += reward
            self.agent.update(observation, reward, done)
        print("Episode finished after {} timesteps".format(t+1))
        self.stats.addEpisode(total_reward)
        self.stats.print()

    def run_episodes(self, n, render = True):
        for i in range(n):
            print("Running episode", i)
            self.run_episode(render)

def quantize(x, n_div, mini, maxi):
    y = (x - mini) / (maxi - mini)
    y = min(0.99, max(0, y))
    return int(y * n_div)

class QLearningAgent(Agent):
    def new_episode(self, observation):
        a_0 = [x[0] for x in self.q]
        a_1 = [x[1] for x in self.q]
        #print((sum(a_0) + sum(a_1)) / (len(a_0) + len(a_1)))
        #print(min(self.q))
        #print(max(self.q))
        #self.q = [[100 for a in range(self.na)] for s in range(self.ns)]

    def __init__(self):

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
        s = self.discretize(obs)
        if done:
            self.q[self.prev_s][self.prev_a] = (1 - self.alpha) * self.q[self.prev_s][self.prev_a] + self.alpha * reward
        else:
            self.q[self.prev_s][self.prev_a] = (1 - self.alpha) * self.q[self.prev_s][self.prev_a] + self.alpha * (reward + self.gamma * max(self.q[s]))

class LinearAgent(Agent):
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

        a = self.X.item(1)
        b = self.measured_X.item(1)
        a = self.X.item(3)
        b = self.measured_X.item(3)

    def update(self, obs, reward, done):
        self.measured_X = np.matrix(obs).T
        self.rectify()

    def new_episode(self, obs):
        self.measured_X = np.matrix(obs).T

class LearningLinearAgent(Agent):
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
        self.measured_X = Y

    def new_episode(self, obs):
        self.measured_X = np.matrix(obs).T
        self.X = None

env = gym.make('CartPole-v0')
#a = LinearAgent()
#a = LearningLinearAgent()
#a = Agent()
#a = RandomCollecting()

a = Controller(env, QLearningAgent())
b = Controller(env, LinearAgent())
c = Controller(env, LearningLinearAgent())
a.run_episodes(5)
b.run_episodes(5)
c.run_episodes(5)
