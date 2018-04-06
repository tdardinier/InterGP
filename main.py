import gym
from controller import *
from agents import qlearning, linear

env = gym.make('CartPole-v0')

a = Controller(env, qlearning.QLearningAgent())
b = Controller(env, linear.LearningLinearAgent())
a.run_episodes(2)
b.run_episodes(5, False)

env.close()
