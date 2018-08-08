import systident.interface as inter
import tensorflow as tf
import gym
import numpy as np
import systident.entropySearch as ES

rtheta = [1.0, 0.1]  # masscart, masspole
theta = [1.0, 0.1]  # masscart, masspole


def modifyEnv(env, theta):
    env.masscart = theta[0]
    env.masspole = theta[1]


def getPolicy(theta):
    senv = inter.createEnv()
    modifyEnv(senv.env.env, theta)
    tf.initialize_all_variables().run()
    pi = inter.learn(senv)
    tf.get_variable_scope().reuse_variables()
    return pi


def collectRealTrajectory(pi):

    env = gym.make('CartPole-v1')
    observation = env.reset()
    obs = [observation]
    actions = []
    t = 0
    while True:
        env.render()
        act = pi.act(False, observation)[0]
        observation, reward, done, info = env.step(act)
        obs.append(observation)
        actions.append(act)
        t += 1
        if done:
            break
    print("TOTAL TIME REAL SIMULATION", t)
    return actions, obs


def generateErrorFunction(actions, observations):

    def evaluateError(theta):

        def norm(a, b):
            return np.linalg.norm(b - a)

        env = gym.make('CartPole-v1')
        modifyEnv(env.env, theta)
        s = .0

        for i in range(len(actions)):

            act = actions[i]

            before, after = observations[i], observations[i+1]
            env.reset()
            env.env.state = before

            obs, _, _, _ = env.step(act)

            s += norm(obs, after)

        return s

    return evaluateError


def findMinimum(f, bounds):

    d = len(bounds)
    es = ES.EntropySearch(d=d, bounds=bounds)

    initial_k = 5
    for _ in range(initial_k):
        x = []
        for j in range(d):
            a, b = bounds[j]
            x.append((b - a) * np.random.random() + a)
        es.addObs(x, f(x))

    current_prob = .0
    while current_prob < 0.9:
        x, c_min, current_prob, _ = es.iterate()
        es.addObs(x, f(x))

    print("Minimum", c_min)

    return es


def iterate(current_theta):

    # Train the policy given our current theta
    pi = getPolicy(current_theta)

    # Collect real trajectory
    actions, observations = collectRealTrajectory(pi)

    # Create error function
    f = generateErrorFunction(actions, observations)
    bounds = [(0.1, 2.), (0.0, 1)]

    es = findMinimum(f, bounds)

    return es
