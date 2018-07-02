from misc import agent
from baselines import deepq


class DeepQ(agent.Agent):

    def __init__(self, env_wrapper):
        self.env_name = env_wrapper.name
        self.env = env_wrapper.env
        self.name = "DeepQ"
        self.model = deepq.load("models/" + self.env_name + ".pkl")

    def act(self, obs):
        return self.model(obs[None])[0]
