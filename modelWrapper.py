import numpy as np
import agent
from predictors import linearPredictor, fullPredictor
import replayBuffer as rb

use_full = True
train = False


class modelWrapper(agent.Agent):

    def __init__(self, agent, n=4, m=1,
                 filename="undefined", use_full=True, train=False):

        self.agent = agent

        self.n = n
        self.m = m

        self.use_full = use_full
        self.train = train

        self.buf = rb.ReplayBuffer()
        self.filename = filename

        self.p = linearPredictor.Predictor(n=n, m=m)
        if use_full:
            self.p = fullPredictor.predictor(n=n, m=m)
        self.X = None
        self.U = None
        self.Y = None
        self.diff = []

        self.n_episodes = 0

    def act(self, obs):
        a = self.agent.act(obs)

        self.X.append(np.matrix(obs).T)
        self.U.append(np.matrix(a).T)
        return a

    def predict(self, y):
        x = self.X[-1]
        u = self.U[-1]
        self.buf.addData(x, u, y)
        r = y
        if train:
            r = self.p.evaluate(x, u, y)
        return r

    def update(self, obs, reward, done):
        self.agent.update(obs, reward, done)

        y = np.matrix(obs).T
        self.Y.append(y)
        yy = self.predict(y)

        if train:
            delta = sum(abs(y - yy)) / self.n
            print("Delta:", delta)
            self.diff.append(delta)
            li = self.diff[-100:]
            print("Average:", sum(li) / len(li))

    def end_episode(self, score):
        self.agent.end_episode(score)

        self.p.addData(self.X, self.U, self.Y)
        # self.p.printer(self.X[0])
        self.n_episodes += 1

        if self.n_episodes % 10 == 0:
            print("# episodes:", self.n_episodes)
            self.buf.save(self.filename)
            if train:
                self.p.train()

    def new_episode(self, obs):
        self.agent.new_episode(obs)

        self.X = []
        self.U = []
        self.Y = []
