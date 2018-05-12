import gym
import tools
import controller as ctrl
from agents import linear, random
from modelWrapper import ModelWrapper
import numpy as np
import random as rd
from evaluator import Evaluator
from result import Result
from visualisator import Visualisator
from predictors import gaussianProcesses, linearPredictor, fullPredictor

default_n_steps = 10000
default_c = 100
default_k = 10
default_env_name = "CartPole-v1"
default_render = True
default_agent = random.Random
default_agent_name = "random"
default_class_predictor = gaussianProcesses
default_predictor_name = "GP"

# Classic Control
acrobot = "Acrobot-v1"
cartpole = "CartPole-v1"
moutain_car = 'MountainCar-v0'
moutain_car_continuous = 'MountainCarContinuous-v0'
pendulum = 'Pendulum-v0'

classic_control = [acrobot, cartpole, moutain_car,
                   moutain_car_continuous, pendulum]

# MuJoCo
ant = "Ant-v2"
cheetah = "HalfCheetah-v2"
hopper = "Hopper-v2"
humanoid = "Humanoid-v2"
humanoid_standup = "HumanoidStandup-v2"
double_pendulum = "InvertedDoublePendulum-v2"
pendulum = "InvertedPendulum-v2"
reacher = "Reacher-v2"
swimmer = "Swimmer-v2"
walker = "Walker2d-v2"

mujoco = [ant, cheetah, hopper, humanoid_standup, double_pendulum,
          pendulum, reacher, swimmer, walker]

predictors = [linearPredictor, fullPredictor, gaussianProcesses]


def collect_1(
    env_name=default_env_name,
    n_steps=default_n_steps,
    render=default_render,
    policy=default_agent,
):
    env = gym.make(env_name)

    m = 1
    for x in env.action_space.shape:
        m *= x
    agent = policy(env.action_space.sample)

    a = ModelWrapper(env_name, agent, env.observation_space.shape[0], m)
    c = ctrl.Controller(env, a)
    c.run_episodes(n_episodes=None, n_steps=n_steps, render=render)

    env.close()


def collectAll(names=classic_control, n_steps=default_n_steps):
    for name in names:
        collect_1(env_name=name, n_steps=n_steps, render=False)


def evaluate_2(
    classPredictor=default_class_predictor,
    agent_name=default_agent_name,
    env_name=cartpole,
    c=default_c,
    k=default_k,
):
    ev = Evaluator(classPredictor, env_name, agent_name)
    r = ev.crossValidate(c=c, k=k)
    return r


def evaluateAllPredictors(
    predictors=predictors,
    agent_name=default_agent_name,
    env_name=cartpole,
    c=default_c,
    k=default_k,
):

    results = []
    for predictor in predictors:
        print("Main: Evaluating", predictor)
        r = evaluate_2(predictor, agent_name, env_name, c, k)
        results.append(r)
    return results


def visualize_3(
    predictor_name=default_predictor_name,
    env_name=default_env_name,
    agent_name=default_agent_name,
    c=default_c,
):

    filename = tools.FileNaming.resultName(
        predictor_name, env_name, agent_name, c)
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
