import gym
from controller import *
from agents import learning, linear

env = gym.make('CartPole-v0')

a = learning.LearningAgent(learning_type = learning.LearningType.MONTECARLO)
ca = Controller(env, a)
b = Controller(env, linear.LearningLinearAgent())
ca.run_episodes(1000, False)
#ca.exploration_type = learning.ExplorationType.OPTIMAL
ca.run_episodes(1000, True)
b.run_episodes(5, False)

env.close()
