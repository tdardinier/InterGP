from interGP import InterGP


class InterGPPlus:

    def __init__(self, k, m=1, sigma=None, dependent=False):

        self.GPs = [InterGP(k) for _ in range(m)]
        self.dependent = dependent
        self.m = m

        self.sigma = sigma
        if sigma is None:
            self.sigma = [i for i in range(m)]

    def fit(self, X, Y):

        XX = X

        for i in range(self.m):

            dim = self.sigma[i]
            YY = [y[dim] for y in Y]
            self.GPs[dim].fit(XX, YY)

            if self.dependent:
                XX = [XX[j] + [YY[j]] for j in range(len(XX))]

    def predictSingle(self, x):

        xx = x
        y = [None for _ in range(self.m)]
        delta = [None for _ in range(self.m)]

        for i in range(self.m):

            dim = self.sigma[i]
            yy, dd = self.GPs[dim].predictSingle(xx)
            y[dim] = yy.item(0)
            delta[dim] = dd.item(0)

            if self.dependent:
                xx = xx + [y[dim]]

        return y, delta

    def predictState(self, bounds, p=0.95):

        xx = bounds
        y = [None for _ in range(self.m)]

        for i in range(self.m):

            dim = self.sigma[i]
            y[dim] = self.GPs[dim].predictState(xx, p)

            if self.dependent:
                xx = xx + [y[dim]]

        return y
