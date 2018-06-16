# from predictors import gaussianProcesses, linearPredictor, \
#    fullPredictor, identity
from predictors import gaussianProcesses, \
    identity
# from agents import random, deepQ
from agents import random
import gym


class EnvWrapper:

    def __init__(self, name, n=4, m=1, aim=499):
        self.name = name
        self.n = n
        self.m = m
        self.env = None

    def make(self):
        if self.env is None:
            self.env = gym.make(self.name)

    def close(self):
        self.env.close()


class PredictorWrapper:

    def __init__(self, predictor, name):
        self.predictor = predictor
        self.name = name


class AgentWrapper:

    def __init__(self, agent, name):
        self.agent = agent
        self.name = name


# Classic Control
acrobot = EnvWrapper("Acrobot-v1", 6, 1)
cartpole = EnvWrapper("CartPole-v1", 4, 1, 499)
moutain_car = EnvWrapper('MountainCar-v0', 2, 1)
moutain_car_continuous = EnvWrapper('MountainCarContinuous-v0', 2, 1)
pendulum = EnvWrapper('Pendulum-v0', 3, 1)

classic_control = [acrobot, cartpole, moutain_car,
                   moutain_car_continuous, pendulum]

# MuJoCo
ant = EnvWrapper("Ant-v2", 111, 8)
cheetah = EnvWrapper("HalfCheetah-v2", 17, 6)
hopper = EnvWrapper("Hopper-v2", 11, 3)
humanoid = EnvWrapper("Humanoid-v2", 376, 17)
humanoid_standup = EnvWrapper("HumanoidStandup-v2", 376, 17)
double_pendulum = EnvWrapper("InvertedDoublePendulum-v2", 11, 1)
inverted_pendulum = EnvWrapper("InvertedPendulum-v2", 4, 1)
reacher = EnvWrapper("Reacher-v2", 11, 2)
swimmer = EnvWrapper("Swimmer-v2", 8, 2)
walker = EnvWrapper("Walker2d-v2", 17, 6)

mujoco = [ant, cheetah, hopper, humanoid, humanoid_standup,
          double_pendulum, inverted_pendulum, reacher, swimmer, walker]

owned = list(classic_control)
owned.append(double_pendulum)
owned.append(inverted_pendulum)
owned.append(reacher)

hard = list(set(classic_control + mujoco) - set(owned))

# linear_predictor = PredictorWrapper(linearPredictor.Predictor, "linearNN")
# full_predictor = PredictorWrapper(fullPredictor.Predictor, "fullNN")
gp = PredictorWrapper(gaussianProcesses.Predictor, "GP")
ngp = PredictorWrapper(gaussianProcesses.Predictor, "NGP")
fngp = PredictorWrapper(gaussianProcesses.Predictor, "FNGP")
id_predictor = PredictorWrapper(identity.Predictor, "identity")

# predictors = [linear_predictor, full_predictor, gp, id_predictor]
predictors = [gp, id_predictor]

agent_random = AgentWrapper(random.Random, "random")
agent_acktr = AgentWrapper(None, "acktr")
# deepq = AgentWrapper(deepQ.DeepQ, "deepQ")
deepq = AgentWrapper(None, "deepQ")

default_n_steps = 20000
default_c = 100
default_render = False
default_agent = agent_random
default_predictor = gp
default_env = cartpole
