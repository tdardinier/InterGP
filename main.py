import gym
import controller as ctrl
from modelWrapper import ModelWrapper
from evaluator import Evaluator
from visualisator import Visualisator
import definitions as d
import tools
import replayBuffer


def collect_1(
    env=d.default_env,
    agent=d.default_agent,
    n_steps=d.default_n_steps,
    render=d.default_render,
):
    nenv = gym.make(env.name)

    m = 1
    for x in nenv.action_space.shape:
        m *= x
    policy = agent.agent(nenv.action_space.sample)

    a = ModelWrapper(env.name, policy, nenv.observation_space.shape[0], m)
    c = ctrl.Controller(nenv, a)
    c.run_episodes(n_episodes=None, n_steps=n_steps, render=render)

    nenv.close()


def collectAll(
    envs=d.classic_control,
    agents=[d.default_agent],
    n_steps=d.default_n_steps,
    render=d.default_render,
):
    for env in envs:
        for agent in agents:
            collect_1(env=env, agent=agent, n_steps=n_steps, render=render)


def evaluate_2(
    predictor=d.default_predictor,
    agent=d.default_agent,
    env=d.default_env,
    c=d.default_c,
):
    ev = Evaluator(predictor.predictor, env.name, agent.name)
    # r = ev.crossValidate(c=c, k=k)
    r = ev.sampleValidate(c=c)
    return r


def evaluateAll(
    predictors=d.predictors,
    envs=d.classic_control,
    agents=[d.default_agent],
    cs=[d.default_c],
):

    results = []
    for c in cs:
        for agent in agents:
            for env in envs:
                for predictor in predictors:
                    print("Main: Evaluating", predictor.name, "in",
                          env.name, "c =", c)
                    r = evaluate_2(predictor, agent, env, c)
                    results.append(r)
    return results


def visualize_3(
    predictors=d.predictors,
    envs=d.classic_control,
    agents=[d.default_agent],
    cs=[d.default_c],
):

    v = Visualisator()
    v.compare(
        [p.name for p in predictors],
        envs,
        [a.name for a in agents],
        cs
    )


def visualizeSigma(
    env=d.default_env,
    agent=d.default_agent,
    c=d.default_c,
):
    v = Visualisator()
    v.plotSigma(env.name, agent_name=agent.name, c=c)


def getReplayBuffer(env=d.default_env, agent=d.default_agent):
    f = tools.FileNaming.replayName(env.name, agent.name)
    return replayBuffer.ReplayBuffer(filename=f)
