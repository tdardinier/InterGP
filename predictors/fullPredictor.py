import tensorflow as tf
import numpy as np
import random as rd
import math

nrd = np.random


class predictor():

    def randomVar(self, n = 1, m = 1, name = "Undefined"):
        init = tf.random_uniform(shape=[n, m])
        return tf.Variable(init)

    def createHidden(self, inp, a, b, name = "Undefined"):
        W = self.randomVar(b, a, name=name + "W")
        B = self.randomVar(b, 1, name=name + "B")
        H = tf.nn.relu(W @ inp + B, name=name + "H")
        return W, B, H

    def createNetwork(self):

        def addHidden(w, b, h):
            self.hiddenW.append(w)
            self.hiddenB.append(b)
            self.hiddenH.append(h)

        self.X = tf.placeholder("float", shape=(None, self.n, 1), name="X")
        self.U = tf.placeholder("float", shape=(None, self.m, 1), name="U")
        self.Y = tf.placeholder("float", shape=(None, self.n, 1), name="Y")

        # self.rX = tf.reshape(self.X, [self.n, -1])
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

        (w, b, h) = self.createHidden(self.x, somme, self.k, "hidden0")
        addHidden(w, b, h)
        inp = h

        for i in range(self.n_layers - 1):
            (w, b, h) = self.createHidden(inp, self.k, self.k, "hidden" + str(i + 1))
            inp = h
            addHidden(w, b, h)

        self.w = self.randomVar(self.n, self.k, name="W")
        self.b = self.randomVar(self.n, 1, name="B")
        self.output = self.w @ inp + self.b

        self.cost = tf.reduce_mean(tf.pow(self.output - self.rY, 2), name="Cost")

        self.optimizer = tf.train.AdamOptimizer().minimize(self.cost)

        # Initialize the variables (i.e. assign their default value)
        init = tf.global_variables_initializer()

        self.sess = tf.Session()

        self.sess.run(init)

    def __init__(self, n = 2, m = 1, k = 20, n_layers = 7):

        self.BATCH_SIZE = 64
        self.use_squared = False

        self.n = n
        self.m = m
        self.k = k
        self.n_layers = n_layers

        self.data_X = []
        self.data_U = []
        self.data_Y = []

        self.createNetwork()

    def addData(self, x, u, y):
        self.data_X += x
        self.data_U += u
        self.data_Y += y

    def addElement(self, x, u, y):
        self.data_X.append(x)
        self.data_U.append(u)
        self.data_Y.append(y)

    def train(self, n_epochs = 1000, n_max = 10000):

        n = len(self.data_X)
        indices = [i for i in range(n)]
        n = min(n, (n_max // self.BATCH_SIZE) * self.BATCH_SIZE)
        n_batchs = n // self.BATCH_SIZE
        print("n", n, ", n_batchs", n_batchs)

        for epoch in range(n_epochs):

            rd.shuffle(indices)
            cost = 0

            for batch in range(n_batchs):

                a = batch * self.BATCH_SIZE
                b = (batch + 1) * self.BATCH_SIZE
                r = range(a, b)
                x = np.asarray([self.data_X[i] for i in r])
                u = np.asarray([self.data_U[i] for i in r])
                y = np.asarray([self.data_Y[i] for i in r])

                d = {self.X: x, self.U: u, self.Y: y}

                _, c = self.sess.run([self.optimizer, self.cost], feed_dict=d)
                cost += math.sqrt(c) / n_batchs

            print("EPOCH:", epoch, ", COST:", cost, "BATCH_SIZE:", self.BATCH_SIZE)

    def evaluate(self, x, u, y):
        d = {self.X: [x], self.U: [u], self.Y: [y]}
        return self.sess.run(self.output, feed_dict=d)


# p = Predictor(2)
# (x, u, y) = generateData()
# p.addData(x, u, y)
