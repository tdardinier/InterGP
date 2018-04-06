import random as rd
import agent
import tools
from enum import Enum

class ExplorationType(Enum):
    OPTIMAL = 0
    EPSILON = 1
    SOFTMAX = 2

class LearningType(Enum):
    NOLEARNING = 0
    QLEARNING = 1
    #SARSA = 2
    MONTECARLO = 3

class LearningAgent(agent.Agent):

    def __init__(self,
                 epsilon = 0.1,
                 alpha = 0.1,
                 gamma = 0.99,
                 tau = 1,
                 exploration_type = ExplorationType.SOFTMAX,
                 learning_type = LearningType.QLEARNING):

        self.states = []
        self.actions = []
        self.rewards = []

        self.n_div = 4
        self.mini = [-0.2, -1, -0.25, -2]
        self.maxi = [0.2, 1, 0.25, 2]

        self.ns = self.n_div ** 4
        self.na = 2

        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma
        self.tau = tau

        self.exploration_type = exploration_type
        self.learning_type = learning_type

        self.prev_s = 0
        self.prev_a = 0

        self.q = [[200 for a in range(self.na)] for s in range(self.ns)]

    def quantize(self, obs):
        return tools.discretize(obs, self.n_div, self.mini, self.maxi)

    def new_episode(self, obs):
        self.states = [self.quantize(obs)]
        self.actions = []
        self.rewards = []

    def end_episode(self):
        if self.learning_type == LearningType.MONTECARLO:
            csum = 0.0
            for i in range(len(self.rewards) - 1, -1, -1):
                s = self.states[i]
                r = self.rewards[i]
                a = self.actions[i]
                csum = r + self.gamma * csum
                self.q[s][a] = (1 - self.alpha) * self.q[s][a] \
                        + self.alpha * csum

    def act(self, obs):

        s = self.quantize(obs)
        a = None

        if self.exploration_type == ExplorationType.EPSILON:
            if rd.random() < self.epsilon:
                a = rd.randint(0, 1)
            else:
                if self.q[s][0] > self.q[s][1]:
                    a = 0
                else:
                    a = 1
        elif self.exploration_type == ExplorationType.SOFTMAX:
            a = tools.softmax(self.q[s], self.tau)
        else:
            a = np.argmax(self.q[s])

        self.prev_a = a
        self.prev_s = s

        self.actions.append(a)

        return a

    def update(self, obs, reward, done):

        s = self.quantize(obs)

        self.states.append(s)
        self.rewards.append(reward)

        if self.learning_type == LearningType.QLEARNING:
            if done:
                self.q[self.prev_s][self.prev_a] = (1 - self.alpha) * self.q[self.prev_s][self.prev_a] + self.alpha * reward
            else:
                self.q[self.prev_s][self.prev_a] = (1 - self.alpha) * self.q[self.prev_s][self.prev_a] + self.alpha * (reward + self.gamma * max(self.q[s]))

