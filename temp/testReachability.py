import misc.definitions as d
from misc import tools, replayBuffer
from compGP.compGP import CompGP
import numpy as np
from misc.visualisator import Visualisator
from compGP.trajectory import Trajectory
from conf import Conf


def getReplayBuffer(env=d.default_env, agent=d.default_agent):
    f = tools.FileNaming.replayName(env.name, agent.name)
    return replayBuffer.ReplayBuffer(filename=f)


def test(c=200, n_test=50, env=d.default_env,
         agent=d.default_agent, k=10, p=0.9):

    buf = getReplayBuffer(env=env, agent=agent)
    buf = buf.cut(c + n_test)

    buf = buf.normalize()

    test = buf.slice([(0, n_test)])
    train = buf.slice([(n_test, c + n_test)])

    conf = Conf(n=len(buf.x[0]), m=len(buf.u[0]))

    cgp = CompGP(conf)

    def convert(x):
        return list(np.array(x.T)[0])

    X = [convert(xx) for xx in train.x]
    U = [convert(uu) for uu in train.u]
    Y = [convert(yy) for yy in train.y]

    cgp.fit(X, U, Y)

    x_0 = convert(test.x[0])
    U = [convert(uu) for uu in test.u]

    S, P = cgp.synthesizeSets(x_0, U, k, p)

    return Trajectory(S, P, test)


v = Visualisator()
