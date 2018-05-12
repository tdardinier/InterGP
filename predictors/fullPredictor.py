import tensorflow as tf
import numpy as np
from predictors import NNPredictor

nrd = np.random


class Predictor(NNPredictor.Predictor):

    def __init__(self, n=2, m=1, k=20, n_layers=7):
        super().__init__(n=n, k=k, m=m, n_layers=n_layers)
        self.name = "fullNN"

    def __createHidden(self, inp, a, b, name="Undefined"):
        W = self.__randomVar(b, a, name=name + "W")
        B = self.__randomVar(b, 1, name=name + "B")
        H = tf.nn.relu(W @ inp + B, name=name + "H")
        return W, B, H

    def __buildNetwork(self):
        def addHidden(w, b, h):
            self.hiddenW.append(w)
            self.hiddenB.append(b)
            self.hiddenH.append(h)

        self.X = tf.placeholder("float", shape=(None, self.n, 1), name="X")
        self.U = tf.placeholder("float", shape=(None, self.m, 1), name="U")
        self.Y = tf.placeholder("float", shape=(None, self.n, 1), name="Y")

        self.rX = tf.reshape(tf.transpose(self.X, perm=[1, 2, 0]), [self.n, -1])
        self.rU = tf.reshape(tf.transpose(self.U, perm=[1, 2, 0]), [self.m, -1])

        l = [self.rX, self.rU]
        if self.use_squared:
            self.X2 = tf.square(self.rX)
            l.append(self.X2)

        self.x = tf.concat(l, 0)
        self.rY = tf.reshape(tf.transpose(self.Y, perm=[1, 2, 0]), [self.n, -1])

        self.hiddenW = []
        self.hiddenB = []
        self.hiddenH = []

        somme = self.n + self.m
        if self.use_squared:
            somme += self.n

        (w, b, h) = self.__createHidden(self.x, somme, self.k, "hidden0")
        addHidden(w, b, h)
        inp = h

        for i in range(self.n_layers - 1):
            (w, b, h) = self.__createHidden(inp, self.k, self.k, "hidden" + str(i + 1))
            inp = h
            addHidden(w, b, h)

        self.w = self.__randomVar(self.n, self.k, name="W")
        self.b = self.__randomVar(self.n, 1, name="B")
        self.output = self.w @ inp + self.b

        self.cost = tf.reduce_mean(tf.pow(self.output - self.rY, 2), name="Cost")

        self.optimizer = tf.train.AdamOptimizer().minimize(self.cost)

        # Initialize the variables (i.e. assign their default value)
        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)
