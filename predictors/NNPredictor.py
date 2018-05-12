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
        self.BATCH_SIZE = 16
        self.use_squared = False

        self.k = k
        self.n_layers = n_layers

        self.n_costs = 100
        self.display_epochs = 200

        self.__buildNetwork()

    def __randomVar(self, n=1, m=1, name="Undefined"):
        init = tf.random_uniform(shape=[n, m])
        return tf.Variable(init)

    def train(self):
        n = len(self.data_X)
        indices = [i for i in range(n)]
        n_batchs = n // self.BATCH_SIZE
        print(self.name + ": n", n, ", n_batchs", n_batchs)

        min_cost = math.inf
        last_cost = [math.inf for _ in range(self.n_costs)]

        epoch = 0
        while min_cost in last_cost:

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

            last_cost[epoch % self.n_costs] = cost
            min_cost = min(min_cost, cost)
            epoch += 1

            if epoch % self.display_epochs == 0:
                print(self.name + ": EPOCH:", epoch,
                      ", COST:", cost, "BATCH_SIZE:", self.BATCH_SIZE)

    def predict(self, x, u):
        d = {self.X: [x], self.U: [u]}
        return self.sess.run(self.output, feed_dict=d)
