import gym
from controller import *
from agents import learning, linear

env = gym.make('CartPole-v0')

p_noise = 0.1

agent_a = learning.LearningAgent(learning_type = learning.LearningType.MONTECARLO)
agent_b = learning.LearningAgent(learning_type = learning.LearningType.QLEARNING)
agent_c = linear.LearningLinearAgent()
a = Controller(env, agent_a)
b = Controller(env, agent_b)
c = Controller(env, agent_c)
#ra = a.run_episodes(1000)
#rb = b.run_episodes(1000)
rc = c.run_episodes(1000)

#env.close()
