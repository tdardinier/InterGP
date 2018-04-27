import gym
import controller as ctrl
from agents import learning, linear

# def car_score(x):
#     return -x.item(0)
#
# car_env = gym.make('MountainCar-v0')
# car_agent = linear.LinearAgent(
#     n=2,
#     actions=[0, 1, 2],
#     score_function=car_score,
# )
# car_ctrl = ctrl.Controller(car_env, car_agent)
# car_results = car_ctrl.run_episodes(5)
# car_env.close()



def cartpole_score(x):
    alpha = 0.5
    xx = abs(x.item(0) / 2.4)
    yy = abs(x.item(2) / 0.2)
    return xx ** alpha + yy * alpha

cartpole_env = gym.make('CartPole-v1')
cartpole_agent = linear.LinearAgent(
    n=4,
    actions=[-1, 1],
    score_function=cartpole_score,
)
cartpole_ctrl = ctrl.Controller(cartpole_env, cartpole_agent)
cartpole_results = cartpole_ctrl.run_episodes(1000)






p_noise = 0.1

agent_a = learning.LearningAgent(
    learning_type=learning.LearningType.MONTECARLO)
agent_b = learning.LearningAgent(learning_type=learning.LearningType.QLEARNING)
a = ctrl.Controller(env, agent_a)
b = ctrl.Controller(env, agent_b)
# ra = a.run_episodes(1000)
# rb = b.run_episodes(1000)

# env.close()
