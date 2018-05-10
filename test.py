from evaluator import Evaluator
from predictors import linearPredictor, fullPredictor, gaussianProcesses

ev = Evaluator("cartpole")


def linear():
    a = ev.crossValidate(linearPredictor)
    print(a)
    return a


def full():
    b = ev.crossValidate(fullPredictor)
    print(b)
    return b


def GP(k=10, n=1000):
    ev = Evaluator("cartpole", n)
    a = ev.crossValidate(gaussianProcesses, k=k)
    print(a)
    return a
