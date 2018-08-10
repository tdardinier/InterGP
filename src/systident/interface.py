#!/usr/bin/env python3
# noinspection PyUnresolvedReferences
from mpi4py import MPI
from baselines.common.cmd_util import make_mujoco_env
from baselines import logger
from baselines.ppo1.mlp_policy import MlpPolicy
from baselines.trpo_mpi import trpo_mpi


def createEnv(env_id='CartPole-v1', seed=0):
    import baselines.common.tf_util as U
    sess = U.single_threaded_session()
    sess.__enter__()

    rank = MPI.COMM_WORLD.Get_rank()
    if rank == 0:
        logger.configure()
    else:
        logger.configure(format_strs=[])
        logger.set_level(logger.DISABLED)
    workerseed = seed + 10000 * MPI.COMM_WORLD.Get_rank()
    return make_mujoco_env(env_id, workerseed)


def learn(env, n_episodes=200, seed=0):
    def policy_fn(name, ob_space, ac_space):
        return MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
                         hid_size=32, num_hid_layers=2)
    test = trpo_mpi.Test()
    pi = test.learn(env,
                    policy_fn,
                    timesteps_per_batch=1024,
                    max_kl=0.01,
                    cg_iters=10, cg_damping=0.1,
                    gamma=0.99, lam=0.98,
                    vf_iters=5, vf_stepsize=1e-3,
                    max_episodes=n_episodes)
    env.close()
    return pi, test
