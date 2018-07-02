import gym
from misc.agents import linear
import misc.controller as ctrl
import numpy as np
import random as rd


def exampleCartpoleLQR():

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


def examplePendulumLQR():

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


def exampleMoutainContinuousCarLQR():

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
