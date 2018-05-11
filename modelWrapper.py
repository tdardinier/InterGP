import numpy as np
import agent
import replayBuffer as rb


class ModelWrapper(agent.Agent):

    def __init__(self, agent, n=4, m=1, classPredictor=None, filename=None):

        self.agent = agent

        self.n = n
        self.m = m

        self.buf = rb.ReplayBuffer()
        self.filename = filename

        self.p = None
        if classPredictor is not None:
            self.p = classPredictor.Predictor(n=n, m=m)

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
        if self.p is not None:
            r = self.p.evaluate(x, u, y)
        return r

    def update(self, obs, reward, done):
        self.agent.update(obs, reward, done)

        y = np.matrix(obs).T
        self.Y.append(y)
        yy = self.predict(y)

        if self.p is not None:
            delta = sum(abs(y - yy)) / self.n
            print("Delta:", delta)
            self.diff.append(delta)
            li = self.diff[-100:]
            print("Average:", sum(li) / len(li))

    def end_episode(self, score):
        self.agent.end_episode(score)

        if self.p is not None:
            self.p.addData(self.X, self.U, self.Y)
        self.n_episodes += 1

        if self.n_episodes % 10 == 0:
            print("# episodes:", self.n_episodes)
            if self.filename is not None:
                self.buf.save(self.filename)
            if self.p is not None:
                self.p.train()

    def new_episode(self, obs):
        self.agent.new_episode(obs)

        self.X = []
        self.U = []
        self.Y = []
