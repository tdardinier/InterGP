class Agent():
    def __init__(self, env_wrapper):
        self.env_wrapper = env_wrapper

    def new_episode(self, obs):
        pass

    def act(self, obs):
        pass

    def end_episode(self, score):
        pass

    def update(self, obs, reward, done):
        pass

    def save(self):
        pass
