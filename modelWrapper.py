import numpy as np
import agent
import replayBuffer as rb
import tools


class ModelWrapper(agent.Agent):

    def __init__(self, env, agent, classPredictor=None, save_file=True):

        self.env_name = env.name
        self.agent = agent
        self.n = env.env.observation_space.shape[0]
        self.m = tools.getM(env.env)

        self.save_file = save_file

        self.p = None
        if classPredictor is not None:
            self.p = classPredictor.Predictor(n=self.n, m=self.m)

        self.buf = rb.ReplayBuffer()

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
            print("ModelWrapper: Delta", delta)
            self.diff.append(delta)
            li = self.diff[-100:]
            print("ModelWrapper: Average", sum(li) / len(li))

    def end_episode(self, score):
        self.agent.end_episode(score)

        self.n_episodes += 1

        if self.p is not None:
            self.p.addData(self.X, self.U, self.Y)
            if self.n_episodes % 10 == 0:
                self.p.train()

    def new_episode(self, obs):
        self.agent.new_episode(obs)

        self.X = []
        self.U = []
        self.Y = []

    def save(self):
        print("ModelWrapper: Saving #episodes", self.n_episodes)
        if self.save_file:
            f = tools.FileNaming.replayName(self.env_name, self.agent.name)
            self.buf.save(f)
