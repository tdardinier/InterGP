import matplotlib.pyplot as plt
import numpy as np
import matplotlib.mlab as mlab
from scipy.stats import norm


def genBell(mu=0, sigma=1, a=-3.5, b=5, n=1000):
    x = np.linspace(a, b, n)
    return x, mlab.normpdf(x, mu, sigma)


def compareCentered(p=0.9):

    alpha = norm.ppf(0.5 * (1. + p))

    fig, ax = plt.subplots(1, 1)

    c1 = 'red'
    X, Y = genBell()
    plt.plot(X, Y, c=c1, label='Mean = 0, standard deviation = 1')

    inter = 'Confidence interval at ' + str(int(p * 100)) + '%'

    X, Y = genBell(a=-alpha, b=alpha)
    ax.fill_between(X, 0, Y, facecolor=c1, alpha=0.2, label=inter)

    ymax = mlab.normpdf(alpha, 0, 1)
    plt.vlines(x=alpha, color=c1, ymin=0, ymax=ymax)

    c2 = 'blue'
    mu = 2
    X, Y = genBell(mu=mu)
    plt.plot(X, Y, c=c2, label='Mean = 2, standard deviation = 1')

    X, Y = genBell(mu=mu, a=mu-alpha, b=mu+alpha)
    ax.fill_between(X, 0, Y, facecolor=c2, alpha=0.2, label=inter)

    ymax = mlab.normpdf(mu-alpha, mu, 1)
    plt.vlines(x=mu-alpha, color=c2, ymin=0, ymax=ymax)

    plt.axvline(x=-alpha, c='black')
    plt.axvline(x=mu+alpha, c='black')

    plt.legend()

    plt.show()


def compareNotCentered(p=0.9):

    fig, ax = plt.subplots(1, 1)

    mu = 2
    a_tilde = min(norm.ppf(1-p), norm.ppf(1-p, mu))
    b_tilde = max(norm.ppf(p), norm.ppf(p, mu))

    alpha = norm.ppf(0.5 * (1. + p))

    c1 = 'red'
    X, Y = genBell()
    plt.plot(X, Y, c=c1, label='Mean = 0, standard deviation = 1')

    inter = 'Confidence interval at ' + str(int(p * 100)) + '%'

    X, Y = genBell(a=a_tilde, b=b_tilde)
    ax.fill_between(X, 0, Y, facecolor=c1, alpha=0.2, label=inter)

    # ymax = mlab.normpdf(alpha, 0, 1)
    # plt.vlines(x=alpha, color=c1, ymin=0, ymax=ymax)

    c2 = 'blue'
    X, Y = genBell(mu=mu)
    plt.plot(X, Y, c=c2, label='Mean = 2, standard deviation = 1')

    X, Y = genBell(mu=mu, a=a_tilde, b=b_tilde)
    ax.fill_between(X, 0, Y, facecolor=c2, alpha=0.2, label=inter)

    # ymax = mlab.normpdf(mu-alpha, 1.5, 1)
    # plt.vlines(x=mu-alpha, color=c2, ymin=0, ymax=ymax)

    plt.axvline(x=a_tilde, c='green')
    plt.axvline(x=b_tilde, c='green')

    plt.axvline(x=-alpha, c='black')
    plt.axvline(x=mu+alpha, c='black')

    plt.legend()

    print(norm.cdf(b_tilde) - norm.cdf(a_tilde))
    print(norm.cdf(b_tilde, mu) - norm.cdf(a_tilde, mu))

    plt.show()
