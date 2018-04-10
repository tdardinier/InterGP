import gym
import random as rd

class Stats():
    def __init__(self):
        self.stats = []

    def addEpisode(self, total_reward):
        self.stats.append(total_reward)

    def getRollingAverage(self, n = 100):
        r = []
        for i in range(len(self.stats) - n + 1):
            l = self.stats[i:i + n]
            r.append(sum(l) / len(l))
        return r

    def printer(self, n = 100):
        l = self.stats[len(self.stats) - n:]
        print("Current average:", sum(l) / len(l))

class Controller():

    def __init__(self, env, agent):
        self.env = env
        self.agent = agent
        self.stats = Stats()

    def reset_stats(self):
        self.stats = Stats()

    def run_episode(self, render = True, p_noise = 0.0):
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
            if rd.random() < p_noise:
                action = 1 - action
            observation, reward, done, info = self.env.step(action)
            total_reward += reward
            self.agent.update(observation, reward, done)
        print("Episode finished after {} timesteps".format(t+1))
        self.agent.end_episode()
        self.stats.addEpisode(total_reward)
        self.stats.printer()

    def run_episodes(self, n, render = True, p_noise = 0.0):
        for i in range(n):
            print("Running episode", i)
            self.run_episode(render, p_noise = p_noise)
