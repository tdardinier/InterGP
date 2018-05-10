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


def GP(c=1000, k=10, filename="undefined"):
    r = ev.crossValidate(gaussianProcesses, c=c, k=k, filename=filename)
    return r
