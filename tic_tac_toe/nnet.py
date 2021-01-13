"""
Module used to define the neural network used for learning simultaneously :
- The policy pi used for choosing actions
- The value function for each state
"""
import tensorflow as tf
import numpy as np


class TicTacToeNet(tf.keras.Model):
    """
    Definition of the neural net class
    """

    def __init__(self):
        """
        Constructor of the neural net
        """
        super(TicTacToeNet, self).__init__()

        # Computer vision layers
        # self.conv = tf.keras.layers.Conv2D(128, (3, 3), padding="same", activation="relu", name="conv")
        # self.conv = tf.keras.layers.Conv2D(128, (3, 3), padding="same", activation="relu", name="conv")
        self.conv = tf.keras.layers.Conv2D(128, (3, 3), padding="valid", activation="relu", name="conv")

        # Flattening in order to feed to dense layers
        self.flatten = tf.keras.layers.Flatten()

        # Dense layers which learn the best pi and v for given representations of board state
        self.dense = tf.keras.layers.Dense(512, activation='relu', name="dense")

        # Outputting probabilities for each
        self.policy = tf.keras.layers.Dense(9, activation='softmax', name="policy")
        self.value = tf.keras.layers.Dense(1, activation='tanh', name="value")

    def call(self, inputs):
        """
        Forward pass function of the neural network
        :param inputs: the input board state for tic tac toe
        :return: the value of the state and the policy for this particular state
        """
        x = self.conv(inputs)
        x = self.flatten(x)
        x = self.dense(x)
        return self.policy(x), self.value(x)


if __name__ == "__main__":

    nnet = TicTacToeNet()
    nnet.build((1, 3, 3, 1))
    board = tf.constant([[0, 1, -1], [0, 0, 0], [0, 0, 0]], dtype=tf.float32)
    pi, v = nnet.call(board)
    print(nnet.summary())
    print(pi)
    print(v)

    print(np.array2string(np.arange(10)))


