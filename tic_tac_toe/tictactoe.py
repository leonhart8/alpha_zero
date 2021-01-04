"""
Module defining the tic tac toe game
"""
import numpy as np

class TicTacToe():
    """
    Class which defines the tic tac toe game
    """

    def __init__(self):
        """
        Constructor of the tic tac toe game.
        Initializes the board state and the first player to go
        """
        self.state = np.zeros(9).reshape((3, 3))
        self.player = 1

    def reset(self):
        """
        Constructor of the tic tac toe game.
        Initializes the board state and the first player to go
        """
        self.state = np.zeros(9).reshape((3, 3))
        self.player = 1

    def get_state(self):
        """
        Returns the current board state
        :return: numpy array, current board state
        """
        return self.state

    def get_player(self):
        """
        Returns the current player
        :return: int, the current player
        """
        return self.player

    def step(self, action):
        """
        A player places a "X" or an "O" on the tic tac toe board
        :param action: the index of the cell on which to put the marker
        :return: Nothing, as a side effect modifies the board and changes the current player
        """
        if self.is_valid(action):
            self.state[action] = self.get_player()
            self.player = - self.player

    def is_valid(self, action):
        """
        Checks that the chosen action is valid, i.e no other marker is occupying the current position
        :param action: the index of the board on which to put the marker
        :return: Bool, True if the action is valid False otherwise
        """
        return self.state[action] == 0

    def is_over(self):
        """
        Checks if the game is done i.e if a column, diagonal or line is full of "X" or "O"
        :return: True if the game is over, False otherwise
        """
        state = self.get_state()
        any_col = np.any((np.abs(np.sum(state, axis=0)) == 3))
        any_row = np.any((np.abs(np.sum(state, axis=1)) == 3))
        diag = np.abs(np.sum(state.diagonal()))
        anti_diag = np.abs(np.sum(np.fliplr(state).diagonal()))
        return any_row or any_col or diag == 3 or anti_diag == 3


if __name__ == "__main__":

    ttc = TicTacToe()

    ttc.state[0] = np.ones(3)
    print("Testing row full")
    print(ttc.is_over())

    ttc.reset()

    ttc.state[:, 0] = np.ones(3) * (-1)
    print("testing column full")
    print(ttc.is_over())

    ttc.reset()

    ttc.state[0][0] = 1
    ttc.state[1][1] = 1
    ttc.state[2][2] = 1
    print("Test diagonal full")
    print(ttc.is_over())

    ttc.reset()

    ttc.state[0][2] = 1
    ttc.state[1][1] = 1
    ttc.state[2][0] = 1
    print("Anti diagonal full")
    print(ttc.is_over())

