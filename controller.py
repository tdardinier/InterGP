import gym

class Stats():
    def __init__(self):
        self.stats = []

    def addEpisode(self, total_reward):
        self.stats.append(total_reward)

    def print(self, n = 100):
        l = self.stats[len(self.stats) - n:]
        print("Current average:", sum(l) / len(l))

class Controller():
    def __init__(self, env, agent):
        self.env = env
        self.agent = agent
        self.stats = Stats()

    def run_episode(self, render = True):
        observation = self.env.reset()
        self.agent.new_episode(observation)
        t = 0
        done = False
        total_reward = 0
        while not done:
            t += 1
            if render:
                self.env.render()
            action = self.agent.act(observation)
            observation, reward, done, info = self.env.step(action)
            total_reward += reward
            self.agent.update(observation, reward, done)
        self.agent.end_episode()
        print("Episode finished after {} timesteps".format(t+1))
        self.stats.addEpisode(total_reward)
        self.stats.print()

    def run_episodes(self, n, render = True):
        for i in range(n):
            print("Running episode", i)
            self.run_episode(render)
