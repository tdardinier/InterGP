import random as rd
from misc import tools


class Stats():
    def __init__(self):
        self.stats = []

    def addEpisode(self, total_reward):
        self.stats.append(total_reward)

    def getRollingAverage(self, n=100):
        r = []
        for i in range(len(self.stats) - n + 1):
            liste = self.stats[i:i+n]
            r.append(sum(liste) / len(liste))
        return r

    def printer(self, n=100):
        liste = self.stats[len(self.stats) - n:]
        print("Controller: Current average", sum(liste) / len(liste))


class Controller():

    def __init__(self, env, agent):
        self.env = env
        self.agent = agent
        self.stats = Stats()

    def reset_stats(self):
        self.stats = Stats()

    def run_episode(self, render=True, p_noise=0.0):
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
        print("Controller: Episode finished ({} timesteps)".format(t+1))
        self.agent.end_episode(total_reward)
        self.stats.addEpisode(total_reward)
        self.stats.printer()
        return t

    def less_than(self, x, M):
        if M is None:
            return True
        return x < M

    def run_episodes(self, n_episodes=100, n_steps=50000,
                     render=True, p_noise=0.0):
        step = 0
        episode = 0
        while tools.less_than(episode, n_episodes) and \
                tools.less_than(step, n_steps):
            episode += 1
            print("Controller: Running episode", episode)
            step += self.run_episode(render, p_noise=p_noise)

        self.agent.save()
