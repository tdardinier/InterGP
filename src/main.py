import misc.controller as ctrl
from misc.modelWrapper import ModelWrapper
from misc.evaluator import Evaluator
from misc.visualisator import Visualisator
import definitions as d
from misc import tools, replayBuffer, result
from interGP.compGP import CompGP
import numpy as np
from interGP.trajectory import Trajectory
from conf import Conf
# from baselines import deepq
# from baselines.acktr.run_mujoco import train


# ----------------------------------------------------
# --------------------- GETTERS ----------------------
# ----------------------------------------------------


def getReplayBuffer(
    env=d.default_env,
    agent=d.default_agent
):
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


def getTraj(
    c=d.default_c,
    env=d.default_env,
    agent=d.default_agent,
    p=d.default_p,
    conf=None,
):
    if conf is None:
        conf = Conf()
    f = tools.FileNaming.trajName(
        env.name, agent.name, c, p, conf)
    traj = Trajectory()
    traj.load(f)
    return traj


# ----------------------------------------------------
# ------------------- SIMPLE STEPS -------------------
# ----------------------------------------------------


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


def visualizeComp_3(
    predictors=d.default_predictors,
    envs=d.default_envs,
    agents=d.default_agents,
    cs=d.default_cs,
    density=d.default_density,
    bins=d.default_bins,
):

    v = Visualisator()
    v.compare(
        [p.name for p in predictors],
        envs,
        [a.name for a in agents],
        cs,
        density=density,
        bins=bins,
    )


def synthesize_4(
    c=d.default_c,
    env=d.default_env,
    agent=d.default_agent,
    k=d.default_k_prediction,
    p=d.default_p,
    save=d.default_save,
    test_chaos_theory=False,
):

    buf = getReplayBuffer(env=env, agent=agent)
    buf = buf.normalize()

    buf = buf.cut(c + k + 1)

    test = buf.slice([(0, k + 1)])
    train = buf.slice([(k + 1, c + k + 1)])

    conf = Conf(n=len(buf.x[0]), m=len(buf.u[0]))
    conf.test_chaos_theory = test_chaos_theory

    cgp = CompGP(conf)

    def convert(x):
        return list(np.array(x.T)[0])

    X = [convert(xx) for xx in train.x]
    U = [convert(uu) for uu in train.u]
    Y = [convert(yy) for yy in train.y]

    cgp.fit(X, U, Y)

    x_0 = convert(test.x[0])
    U = [convert(uu) for uu in test.u]

    traj = cgp.synthesizeSets(x_0, U, k, p)
    traj.addBuf(test)

    if save:
        f = tools.FileNaming.trajName(
            env.name, agent.name, c, p, conf)
        traj.save(f)

    return traj


def visualizeSets_5(
    cs=d.default_cs,
    env=d.default_env,
    agent=d.default_agent,
    k=d.default_k_visualization,
    ps=d.default_ps,
    colors=d.default_colors_sets,
    components=d.default_components,
    loc=d.default_loc,
    show=True,
    conf=None,
    test_chaos_theory=False,
):

    if conf is None:
        conf = Conf()
        conf.test_chaos_theory = test_chaos_theory

    trajs = []
    for p in ps:
        for c in cs:
            trajs.append(getTraj(c, env, agent, p, conf=conf))
    v = Visualisator()
    v.show = show
    name = tools.FileNaming.descrName(env, agent, c, conf)
    filename = tools.FileNaming.imageTrajName(env.name, agent.name,
                                              c, p, conf, k)
    v.plotCompGP(trajs, colors=colors, name=name, components=components,
                 loc=loc, k=k, filename=filename)


# --------------------------------------------
# -------------- MULTIPLE STEPS --------------
# --------------------------------------------


def collectAll(
    envs=d.default_envs,
    agents=d.default_agents,
    n_steps=d.default_n_steps,
    render=d.default_render,
):
    for env in envs:
        for agent in agents:
            collect_1(env=env, agent=agent, n_steps=n_steps, render=render)


def evaluateAll(
    predictors=d.default_predictors,
    envs=d.default_envs,
    agents=d.default_agents,
    cs=d.default_cs,
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


def synthesizeAll(
    cs=d.default_cs,
    envs=d.default_envs,
    agents=d.default_agents,
    k=d.default_k_prediction,
    ps=d.default_ps,
    save=d.default_save,
    test_chaos_theory=False,
):

    for c in cs:
        for agent in agents:
            for env in envs:
                for p in ps:
                    synthesize_4(c, env, agent, k, p, save, test_chaos_theory)


def visualizeSetsAll(
    cs=d.default_cs,
    envs=d.default_envs,
    agents=d.default_agents,
    k_max=d.default_k_prediction,
    ps=d.default_ps,
    colors=d.default_colors_sets,
    components=d.default_components,
    loc=d.default_loc,
    show=False,
    conf=None,
    test_chaos_theory=False,
):

    for c in cs:
        for env in envs:
            for agent in agents:
                for p in ps:
                    for k in range(1, k_max + 1):
                        visualizeSets_5([c], env, agent, k, [p], colors,
                                        components, loc, show, conf,
                                        test_chaos_theory)


visualizeSetsAll()

# ----------------------------------------------------
# ----------------------- MISC -----------------------
# ----------------------------------------------------

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
