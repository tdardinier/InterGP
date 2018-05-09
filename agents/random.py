import agent


class Random(agent.Agent):

    def __init__(self, choose_action):
        self.choose_action = choose_action

    def act(self, obs):
        return self.choose_action()
