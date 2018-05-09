import gym
import controller as ctrl
from agents import linear, random, modelWrapper
import numpy as np
import random as rd

cartpole = "CartPole-v1"
pendulum = 'Pendulum-v0'
moutain_car_continuous = 'MountainCarContinuous-v0'
moutain_car = 'MountainCar-v0'

reacher = "Reacher-v2"
swimmer = "Swimmer-v2"
hopper = "Hopper-v2"
humanoid = "Humanoid-v2"
ant = "Ant-v2"
cheetah = "HalfCheetah-v2"
double_pendulum = "InvertedDoublePendulum-v2"


def agentRandom(name, N=100000000000, render=True,
                default_state=None, filename="undefined"):
    if default_state is None:
        env = gym.make(name)
    else:
        env = gym.make(name, default_state)
    m = 1
    for x in env.action_space.shape:
        m *= x
    rand = random.Random(env.action_space.sample)
    a = modelWrapper.modelWrapper(rand, env.observation_space.shape[0],
                                  m, filename=filename)
    c = ctrl.Controller(env, a)
    c.run_episodes(N, render=render)


def agentCartpole():

    c_x = 1  # 2.4
    c_theta = 0.2  # 0.2
    Q = np.diag([1. / (c_x ** 2), 0, 1. / (c_theta ** 2), 0])

    def random_action():
        return rd.choice([-1, 1])

    def convert_obs(obs):
        a = np.matrix(obs).T
        a.put(0, a.item(0) - 2)
        return a

    def convert_action(a):
        return (a + 1) // 2

    def convert_control(u):
        if u.item(0) >= 0:
            return 1
        return -1

    env = gym.make('CartPole-v1')
    agent = linear.LinearAgent(
        n=4,
        Q=Q,
        epsilon=0.05,
        N=20,
        convert_action=convert_action,
        convert_control=convert_control,
        convert_obs=convert_obs,
        random_action=random_action,
    )
    ctl = ctrl.Controller(env, agent)
    results = ctl.run_episodes(1000)
    env.close()
    return results


def agentPendulum():

    Q = np.diag([1, 0, 0])

    def random_action():
        return 4 * rd.random() - 2

    def convert_obs(obs):
        a = np.matrix(obs).T
        a.put(0, a.item(0) - 1)
        return a

    def convert_control(u):
        a = u.item(0)
        a = max(-2, min(2, a))
        return a

    env = gym.make('Pendulum-v0')
    agent = linear.LinearAgent(
        n=3,
        Q=Q,
        epsilon=0.05,
        N=20,
        convert_control=convert_control,
        convert_obs=convert_obs,
        random_action=random_action,
    )
    ctl = ctrl.Controller(env, agent)
    results = ctl.run_episodes(1000)
    env.close()
    return results


def moutainContinuousCar():

    def convert_obs(obs):
        a = np.matrix(obs).T
        a.put(0, a.item(0) - 1)
        return a

    Q = np.diag([1., 0])

    def convert_control(u):
        a = u.item(0)
        return a

    env = gym.make('MountainCarContinuous-v0')
    agent = linear.LinearAgent(
        n=2,
        Q=Q,
        epsilon=0.05,
        N=10,
        convert_control=convert_control,
        convert_obs=convert_obs,
    )
    ctl = ctrl.Controller(env, agent)
    results = ctl.run_episodes(1000)
    env.close()
    return results


# pendulum()
# moutainContinuousCar()
# cartpole()

# p_noise = 0.1
# agent_a = learning.LearningAgent(
#    learning_type=learning.LearningType.MONTECARLO)
# agent_b=learning.LearningAgent(learning_type=learning.LearningType.QLEARNING)
# a = ctrl.Controller(env, agent_a)
# b = ctrl.Controller(env, agent_b)
# ra = a.run_episodes(1000)
# rb = b.run_episodes(1000)
# env.close()
