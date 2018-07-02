import tensorflow as tf
import numpy as np
import random as rd
from predictors import NNPredictor

nrd = np.random


class Predictor(NNPredictor.Predictor):

    def __init__(self, n=2, k=30, m=1, n_layers=2):
        super().__init__(n=n, k=k, m=m, n_layers=n_layers)
        self.name = "linearNN"

    def __buildNetwork(self):
        self.X = tf.placeholder("float", shape=(None, self.n, 1), name="X")
        self.U = tf.placeholder("float", shape=(None, self.m, 1), name="U")
        self.Y = tf.placeholder("float", shape=(None, self.n, 1), name="Y")

        self.rX = tf.reshape(tf.transpose(self.X, perm=[1, 2, 0]), [self.n, -1])
        self.rY = tf.reshape(tf.transpose(self.Y, perm=[1, 2, 0]), [self.n, -1])

        self.Wha1 = self.__randomVar(self.k, self.n, name="Wha1")
        self.Bha1 = self.__randomVar(self.k, 1, name="Bha1")
        self.Ha1 = tf.nn.relu(self.Wha1 @ self.rX + self.Bha1, name="Ha1")

        self.Wha2 = self.__randomVar(self.k, self.k, name="Wha2")
        self.Bha2 = self.__randomVar(self.k, 1, name="Bha2")
        self.Ha2 = tf.nn.relu(self.Wha2 @ self.Ha1 + self.Bha2, name="Ha2")

        self.Wa = self.__randomVar(self.n, self.k, name="Wa")
        self.Ba = self.__randomVar(self.n, 1, name="Ba")
        self.A = self.Wa @ self.Ha2 + self.Ba

        self.Whw1 = self.__randomVar(self.k, self.n, name="Whw1")
        self.Bhw1 = self.__randomVar(self.k, 1, name="Bhw1")
        self.Hw1 = tf.nn.relu(self.Whw1 @ self.rX + self.Bhw1, name="Hw1")

        self.Whw2 = self.__randomVar(self.k, self.k, name="Whw2")
        self.Bhw2 = self.__randomVar(self.k, 1, name="Bhw2")
        self.Hw2 = tf.nn.relu(self.Whw2 @ self.Hw1 + self.Bhw2, name="Hw2")

        p = self.n * self.m
        self.Ww = self.__randomVar(p, self.k, name="Wa")
        self.Bw = self.__randomVar(p, 1, name="Ba")
        self.W = self.Ww @ self.Hw2 + self.Bw

        self.rW = tf.reshape(tf.transpose(self.W), [-1, self.n, self.m])
        self.prod = self.rW @ self.U
        self.rProd = tf.reshape(tf.transpose(self.prod, [1, 2, 0]), [self.n, -1], name="rProd")

        self.output = self.A + self.rProd

        # Mean squared error
        self.cost = tf.reduce_mean(tf.pow(self.output - self.rY, 2), name="Cost")

        self.optimizer = tf.train.AdamOptimizer().minimize(self.cost)

        # Initialize the variables (i.e. assign their default value)
        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)
