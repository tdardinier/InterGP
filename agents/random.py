import agent


class Random(agent.Agent):

    def __init__(self, env_wrapper):
        self.choose_action = env_wrapper.env.action_space.sample
        self.name = "random"

    def act(self, obs):
        return self.choose_action()
