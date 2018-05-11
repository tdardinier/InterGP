import gym
import controller as ctrl
from agents import linear, random
import modelWrapper
import numpy as np
import random as rd
from evaluator import Evaluator
from result import Result
from visualisator import Visualisator
from predictors import gaussianProcesses
# from predictors import linearPredictor, fullPredictor

# Classic Control
cartpole = "CartPole-v1"
pendulum = 'Pendulum-v0'
moutain_car_continuous = 'MountainCarContinuous-v0'
moutain_car = 'MountainCar-v0'

# MuJoCo
reacher = "Reacher-v2"
swimmer = "Swimmer-v2"
hopper = "Hopper-v2"
humanoid = "Humanoid-v2"
ant = "Ant-v2"
cheetah = "HalfCheetah-v2"
double_pendulum = "InvertedDoublePendulum-v2"


def collect_1(name=cartpole, N_episodes=1000, render=True,
              policy=random.Random, default_state=None, filename="cartpole"):

    if default_state is None:
        env = gym.make(name)
    else:
        env = gym.make(name, default_state)
    m = 1
    for x in env.action_space.shape:
        m *= x
    agent = policy(env.action_space.sample)
    a = modelWrapper.ModelWrapper(agent, n=env.observation_space.shape[0],
                                  m=m, classPredictor=None, filename=filename)
    c = ctrl.Controller(env, a)
    c.run_episodes(N_episodes, render=render)
    env.close()


def evaluate_2(classPredictor=gaussianProcesses, c=100, k=10,
               replay_filename="cartpole", results_filename="GP_cartpole_100"):
    ev = Evaluator(replay_filename)
    r = ev.crossValidate(classPredictor, c=c, k=k, filename=results_filename)
    return r


def visualize_3(filename="GP_cartpole_100"):
    r = Result(filename=filename)
    v = Visualisator(r)
    v.histo()


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
