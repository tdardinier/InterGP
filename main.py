import controller as ctrl
from modelWrapper import ModelWrapper
from evaluator import Evaluator
from visualisator import Visualisator
import definitions as d
import tools
import replayBuffer
import result
# from baselines import deepq
# from baselines.acktr.run_mujoco import train


def collect_1(
    env=d.default_env,
    agent=d.default_agent,
    n_steps=d.default_n_steps,
    render=d.default_render,
):
    env.make()

    policy = agent.agent(env)

    a = ModelWrapper(env, policy)
    c = ctrl.Controller(env.env, a)
    c.run_episodes(n_episodes=None, n_steps=n_steps, render=render)

    env.close()


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
    density=False,
):

    v = Visualisator()
    v.compare(
        [p.name for p in predictors],
        envs,
        [a.name for a in agents],
        cs,
        density=density,
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


def getResults(
    predictor=d.default_predictor,
    agent=d.default_agent,
    env=d.default_env,
    c=d.default_c,
):
    f = tools.FileNaming.resultName(
        predictor.name,
        env.name,
        agent.name,
        c)
    return result.Result(filename=f)


def reachabilityReacher():
    r = getResults(env=d.reacher, c=200)



# def trainModelDeepQ(env_wrapper, aim=499):
#
#     def callback(lcl, _glb):
#         is_solved = lcl['t'] > 100 and \
#             sum(lcl['episode_rewards'][-101:-1]) / 100 >= aim
#         return is_solved
#
#     env_wrapper.make()
#
#     model = deepq.models.mlp([64])
#     act = deepq.learn(
#         env_wrapper.env,
#         q_func=model,
#         lr=1e-3,
#         max_timesteps=100000,
#         buffer_size=50000,
#         exploration_fraction=0.1,
#         exploration_final_eps=0.02,
#         print_freq=10,
#         callback=callback
#     )
#     filename = tools.FileNaming.modelName(env_wrapper)
#     print("Saving model to " + filename)
#     act.save(filename)
#
#
# def collectACKTR(env_wrapper, steps=1000000):
#     agent = d.agent_acktr
#     f = tools.FileNaming.replayName(env_wrapper.name, agent.name)
#     train(env_wrapper.name, steps, 42, f)
