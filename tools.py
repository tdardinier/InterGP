import math
import random as rd

def quantize(x, n_div, mini, maxi):
    y = (x - mini) / (maxi - mini)
    y = min(0.99, max(0, y))
    return int(y * n_div)

def discretize(obs, n_div, mini, maxi):
    #print(obs)
    s = 0
    m = 1
    for i in range(4):
        x = obs[i]
        s += m * quantize(x, n_div, mini[i], maxi[i])
        m *= n_div
    return s

def proba(v, tau):
    return math.exp(v / tau)

def softmax(values, tau):
    p = [proba(x, tau) for x in values]
    seuil = rd.random() * sum(p)
    s = 0.0
    i = 0
    while s < seuil:
        s += p[i]
        i += 1
    return i - 1
