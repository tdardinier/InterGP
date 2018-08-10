import systident.interface as inter
import tensorflow as tf
import gym
import numpy as np
import systident.entropySearch as ES
import math


class Problem:

    def __init__(self, name, theta, bounds, properties):

        self.name = name
        self.theta = theta
        self.bounds = bounds

        def modifyEnv(env, theta):
            for i, x in enumerate(properties):
                setattr(env, x, theta[i])

        self.modifyEnv = modifyEnv

        def setStateFromObs(env, obs):
            env.reset()
            env.env.state = obs

        self.setStateFromObs = setStateFromObs


cartpole = Problem('CartPole-v1', [3, 1],
                   [(0.1, 3), (0.01, 0.8)],
                   ['masscart', 'masspole'])


pendulum = Problem('Pendulum-v0', [3, 0.5],
                   [(0.5, 5), (0.5, 5)],
                   ['attr_mass', 'attr_length'])


def setStateFromObs(env, obs):
    env.reset()
    cos, sin, thetadot = obs
    theta = math.atan2(sin, cos)
    env.env.state = theta, thetadot


pendulum.setStateFromObs = setStateFromObs


class ES_TRPO:

    def __init__(self, problem):

        self.pi = None
        self.actions = None
        self.observations = None

        self.env_name = problem.name
        self.modifyEnv = problem.modifyEnv
        self.setStateFromObs = problem.setStateFromObs
        self.theta = problem.theta
        self.bounds = problem.bounds
        self.es = ES.EntropySearch(self.bounds)

    def findPolicy(self):
        senv = inter.createEnv(self.env_name)
        self.modifyEnv(senv.env.env, self.theta)
        tf.initialize_all_variables().run()
        self.pi, test = inter.learn(senv)
        # print("LEARNING")
        # test.learn_step()
        tf.get_variable_scope().reuse_variables()

    def collectRealTrajectory(self):

        env = gym.make(self.env_name)
        observation = env.reset()
        self.observations = [observation]
        self.actions = []
        r = 0
        while True:
            env.render()
            act = self.pi.act(False, observation)[0]
            observation, reward, done, info = env.step(act)
            self.observations.append(observation)
            self.actions.append(act)
            r += reward
            if done:
                break
        print("TOTAL REWARD REAL SIMULATION", r)

    def generateErrorFunction(self):

        def evaluateError(theta):

            def norm(a, b):
                return np.linalg.norm(b - a)

            env = gym.make(self.env_name)
            self.modifyEnv(env.env, theta)
            s = .0

            for i in range(len(self.actions)):

                act = self.actions[i]

                before, after = self.observations[i], self.observations[i+1]

                self.setStateFromObs(env, before)

                obs, _, _, _ = env.step(act)

                s += norm(obs, after)

            return s

        return evaluateError

    def iterate(self):

        self.findPolicy()
        self.collectRealTrajectory()
        f = self.generateErrorFunction()

        c_min, prob, gp = self.es.findMinimum(f)

        return f, c_min, prob, gp
