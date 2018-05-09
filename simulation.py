import gym
import controller as ctrl
from agents import learning, linear
import matplotlib.pyplot as plt

env = gym.make('CartPole-v0')

p = 0.5
n = 1000
n_average = 100
render = False

agent_a = learning.LearningAgent(
    learning_type=learning.LearningType.MONTECARLO)
agent_b = learning.LearningAgent(learning_type=learning.LearningType.QLEARNING)
agent_c = linear.LearningLinearAgent()
a = ctrl.Controller(env, agent_a)
b = ctrl.Controller(env, agent_b)
c = ctrl.Controller(env, agent_c)
c.run_episodes(n, render=render, p_noise=p)
a.run_episodes(n, render=render, p_noise=p)
b.run_episodes(n, render=render, p_noise=p)

ya = a.stats.getRollingAverage(n_average)
yb = b.stats.getRollingAverage(n_average)
yc = c.stats.getRollingAverage(n_average)
x = range(len(ya))
plt.plot(x, ya, x, yb, x, yc)
plt.title("p = 0.5")
plt.show()

# env.close()
