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
        self.conv = tf.keras.layers.Conv2D(128, (3, 3), padding="valid", activation="relu", name="conv")

        # Flattening in order to feed to dense layers
        self.flatten = tf.keras.layers.Flatten()

        # Dense layers which learn the best pi and v for given representations of board state
        self.dense = tf.keras.layers.Dense(512, activation='relu', name="dense")

        # Outputting probabilities for each
        self.policy = tf.keras.layers.Dense(9, activation='softmax')
        self.value = tf.keras.layers.Dense(1, activation='tanh')

    def call(self, inputs):
        """
        Forward pass function of the neural network
        :param inputs: the input board state for tic tac toe
        :return: the value of the state and the policy for this particular state
        """
        x = tf.reshape(inputs, (1, 3, 3, 1))
        x = self.conv(x)
        x = self.flatten(x)
        x = self.dense(x)
        return self.policy(x), self.value(x)


if __name__ == "__main__":

    nnet = TicTacToeNet()
    board = tf.constant([[0, 1, -1], [0, 0, 0], [0, 0, 0]], dtype=tf.float32)
    pi, v = nnet.call(board)
    print(pi)
    print(v)


