"""
Module used to define the neural network used for learning simultaneously :
- The policy pi used for choosing actions
- The value function for each state
"""
import tensorflow as tf


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
        self.conv1 = tf.keras.layers.Conv2D(128, (3, 3), padding="valid", activation="relu", name="conv1")
        self.conv2 = tf.keras.layers.Conv2D(128, (3, 3), padding="valid", activation="relu", name="conv2")
        self.conv3 = tf.keras.layers.Conv2D(128, (3, 3), padding="valid", activation="relu", name="conv3")

        # Flattening in order to feed to dense layers
        self.flatten = tf.keras.layers.Flatten()

        # Dense layers which learn the best pi and v for given representations of board state
        self.dense = tf.keras.layers.Dense(512, activation='relu', name="dense")

        # Outputting probabilities for each
        self.policy = tf.keras.layers.Dense(9, activation='softmax')
        self.value = tf.keras.layers.Dense(1, activation='tanh')

    def call(self, input):
        """
        Forward pass function of the neural network
        :param input:
        :return:
        """
