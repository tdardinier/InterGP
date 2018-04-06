import random as rd
import agent
from tools import *

class QLearningAgent(agent.Agent):
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
