from oldInterGP.interval import Interval
import numpy as np


class InterFunction:

    def __init__(self, f, extrema=[]):
        self.f = f
        self.extrema = extrema  # local extrema

    def image(self, inter):
        xs = [inter.a, inter.b]
        xs += [x for x in self.extrema if inter.contains(x)]
        ys = [self.f(x) for x in xs]
        m = min(ys)
        M = max(ys)
        return Interval(m, M)


square = InterFunction(
    lambda x: x**2,
    [0]
)

exp = InterFunction(np.exp)
