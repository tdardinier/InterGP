import tensorflow as tf
import numpy as np
import random as rd
import predictor
import math

nrd = np.random


# VIRTUAL CLASS
class Predictor(predictor.Predictor):

    def __init__(self, n=2, k=30, m=1, n_layers=5):
        super().__init__(n, m)

        self.name = "virtualNN"
        self.BATCH_SIZE = 32
        self.use_squared = False

        self.k = k
        self.n_layers = n_layers

        self.__buildNetwork()

    def __randomVar(self, n=1, m=1, name="Undefined"):
        init = tf.random_uniform(shape=[n, m])
        return tf.Variable(init)

    def train(self, n_epochs=500, n_max=10000):
        n = len(self.data_X)
        indices = [i for i in range(n)]
        n = min(n, (n_max // self.BATCH_SIZE) * self.BATCH_SIZE)
        n_batchs = n // self.BATCH_SIZE
        print(self.name + ": n", n, ", n_batchs", n_batchs)

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

            if epoch % 50 == 0:
                print(self.name + ": EPOCH:", epoch,
                      ", COST:", cost, "BATCH_SIZE:", self.BATCH_SIZE)

    def predict(self, x, u):
        d = {self.X: [x], self.U: [u]}
        return self.sess.run(self.output, feed_dict=d)
