from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.gaussian_process.kernels import WhiteKernel
from sklearn.gaussian_process.kernels import Matern
import numpy as np
from conf import Conf


class CoreGP():

    def __init__(self, conf=None):

        if conf is None:
            conf = Conf()

        self.scipy = conf.scipy

        if self.scipy:
            self.kernel = C(1.0, (1e-3, 1e3)) * RBF(1, (1e-2, 1e2))
            if conf.matern:
                self.kernel = Matern(length_scale=2, nu=3/2)
            if conf.noise:
                self.kernel += WhiteKernel()

        else:
            self.k = conf.k  # kernel function
            self.X = None
            self.Y = None
            self.N = None

    def train(self, X, Y):

        assert (len(X) == len(Y)), "Wrong sizes X Y"

        if self.scipy:
            self.gp = GaussianProcessRegressor(
                # kernel=self.kernel, n_restarts_optimizer=9, normalize_y=True)
                kernel=self.kernel, n_restarts_optimizer=9)
            self.gp.fit(X, Y)

        else:
            self.X = X
            self.Y = Y
            self.N = len(self.Y)
            self.__preCompute()

    def __generateMatrixCov(self, X1, X2):
        m = []
        for x1 in X1:
            line = []
            for x2 in X2:
                line.append(self.k(x1, x2))
            m.append(line)
        return np.matrix(m)

    def __preCompute(self):
        f = np.matrix(self.Y).reshape((self.N, 1))
        K = self.__generateMatrixCov(self.X, self.X)  # K(X, X)
        self.B = np.linalg.inv(K)
        self.A = self.B * f

    def predict(self, x, return_cov=False):

        if self.scipy:
            if return_cov:
                return self.gp.predict(x, return_cov=True)
            else:
                return self.gp.predict(x, return_std=True)

        else:
            K_star = self.__generateMatrixCov(x, self.X)
            MU = K_star * self.A
            SIGMA = self.__generateMatrixCov(x, x) - K_star * self.B * K_star.T
            return MU, SIGMA

    def preComputeGiven(self, x):
        self.MU, self.SIGMA = self.predict(x, return_cov=True)

    def predictGiven(self, i, yi):

        k = len(self.MU)
        l = [j for j in range(k) if j != i]

        MU_1 = self.MU[np.ix_(l)].reshape([k - 1, 1])
        MU_2 = self.MU[np.ix_([i])]

        SIGMA_11 = self.SIGMA[np.ix_(l, l)]
        SIGMA_12 = self.SIGMA[np.ix_(l, [i])]
        SIGMA_21 = self.SIGMA[np.ix_([i], l)]
        SIGMA_22 = self.SIGMA[np.ix_([i], [i])]

        prod = SIGMA_12 * np.linalg.inv(SIGMA_22)

        mu = MU_1 + prod * (yi - MU_2)
        sigma = SIGMA_11 - prod * SIGMA_21

        return mu, sigma
