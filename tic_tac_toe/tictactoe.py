"""
Module defining the tic tac toe game
"""
import numpy as np


class TicTacToe:
    """
    Class which defines the tic tac toe game
    """

    def reset(self):
        """
        Constructor of the tic tac toe game.
        Initializes the board state and the first player to go
        """
        return np.zeros(9).reshape((3, 3)), 1

    def step(self, state, action, player):
        """
        A player places a "X" or an "O" on the tic tac toe board
        :param state: the current state of the game
        :param action: the index of the cell on which to put the marker
        :param player: the player making the action
        :return: numpy array state and a boolean flag which determines if the game is over or not
        """
        valid_actions = self.valid_actions(state)
        if valid_actions[action]:
            new_state = np.copy(state)
            new_state = new_state.flatten()
            new_state[action] = player
            new_state = new_state.reshape(3, 3)
            return new_state, -player
        else:
            print(state)
            print(action)
            print("Invalid actions")

    def action_space(self):
        """
        Gives the size of the action space i.e number of actions
        :return: int, the number of possible actions
        """
        return 9

    def valid_actions(self, state):
        """
        Returns all the valid actions from a given state
        :return: array, integers of legal actions
        """
        legal_moves = np.zeros(self.action_space())
        for i in range(3):
            for j in range(3):
                if state[i][j] == 0:
                    legal_moves[i*3 + j] = 1
        return legal_moves

    def is_valid(self, state, action):
        """
        Returns True if an action can be done, False otherwise
        :param state: The current state of the tic tac toe game
        :param action: the action to be taken
        :return: True if an action can be done, False otherwise
        """
        return self.valid_actions(state)[action]

    def player_win(self, state, player):
        """
        Checks if the given player won for a given state
        :param state: the current state of the game
        :param player: the player that is being checked
        :return: True if the player won, False otherwise
        """
        win = player * 3

        # Check rows
        for i in range(3):
            count = 0
            for j in range(3):
                count += state[i][j]
            if count == win:
                return True

        # Check cols
        for i in range(3):
            count = 0
            for j in range(3):
                count += state[j][i]
            if count == win:
                return True

        # Check diag
        count = sum(state[i][i] for i in range(3))
        if count == win:
            return True

        # Check anti diag
        count = sum(state[i][j] for i, j in zip(range(0, 3), range(2, -1, -1)))
        if count == win:
            return True

        return False

    def is_terminal(self, state):
        """
        Checks if a given state is terminal
        :param state: state of the tic tac toe game
        :return: True if the game state is terminal, False otherwise
        """
        if self.player_win(state, 1) or self.player_win(state, -1):
            return True

        for i in range(3):
            for j in range(3):
                if state[i][j] == 0:
                    return False
        return True

    def reward(self, state, player):
        """
        Returns the reward associated to the state
        :param state: given a state, checks if the current player won
        :param player: the player whose turn it is
        :return: int, the reward associated to a given state:
        - 0 if the state is non terminal
        - 1 if the current player wins
        - -1 if the current player loses
        """
        # If the game is not over no reward
        if not self.is_terminal(state):
            return 0

        # Positive reward to backup if the current player won
        if self.player_win(state, player):
            return 1

        # If the other player won, negative reward
        if self.player_win(state, -player):
            return -1

        # Tie case
        return 0

    def hash_state(self, state):
        """
        Returns a unique representation of a state used to add it to dictionaries
        :param state: current state
        :return: hashed version of a state
        """
        return np.array2string(state)

    def print_state(self, state):
        """
        Returns a string representation of a state that can be
        :param state:
        :return:
        """
        int_to_repr = {
            1: "X",
            0: " ",
            -1: "O"
        }
        line = "+---+---+---+"
        for i in range(3):
            print(line)
            print("| ", end="")
            for j in range(3):
                print(int_to_repr[state[i][j]], " | ", sep="", end="")
            print()
        print(line)


if __name__ == "__main__":
    ttc = TicTacToe()
    print(ttc.print_state((-1) * np.ones(9).reshape(3, 3)))
    print("hello")

