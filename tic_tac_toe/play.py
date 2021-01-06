"""
Module which defines the Play class. This class is responsible for implementing games between :
- human and ai
- ai and ai
"""
import numpy as np
from tensorflow import keras
from mcts import MCTS
from tictactoe import TicTacToe


class Play:

    def ai_vs_ai(self, model_1, model_2, num_sim=50, verbose=False):
        """
        Plays a game of tic tac toe between two AIs,
        :param model_1: keras model of the first AI
        :param model_2: keras model of the second AI
        :param num_sim: the number of simulations per MCTS
        :return: int, 0 if model 1 loses, 1 if it wins,
        """
        # Initializing the game
        game = TicTacToe()

        # Initializing MCTS
        mcts = MCTS()

        # Getting the first player to go and the initial empty tic tac toe board state
        state, player = game.reset()

        if verbose:
            game.print_state(state)

        # If 1 human goes first else machine goes first
        first_to_go = np.random.randint(2)

        if first_to_go:
            ai_1, ai_2 = 1, -1
        else:
            ai_1, ai_2 = -1, 1

        while not game.is_terminal(state):
            if player == ai_1:
                if verbose:
                    print("Model 1 is currently making a choice")
                for _ in range(num_sim):
                    mcts.search(state, game, player, model_1)
                pi = mcts.best_policy(state, game, tau=0.01)
                action = np.random.choice(game.action_space(), p=pi)
            else:
                if verbose:
                    print("Model 2 is currently making a choice")
                for _ in range(num_sim):
                    mcts.search(state, game, player, model_2)
                pi = mcts.best_policy(state, game, tau=0.01)
                action = np.random.choice(game.action_space(), p=pi)
            state, player = game.step(state, action, player)

            if verbose:
                game.print_state(state)

        if verbose:
            if game.player_win(state, ai_1):
                print("ai 1 won !")
            elif game.player_win(state, ai_2):
                print("ai 2 won !")
            else:
                print("tie !")

        return game.player_win(state, ai_1)

    def human_vs_ai(self, pathname, num_sim=50):
        """
        Plays a game of tic tac toe between an AI trained by AlphaZero and a human player
        :param pathname: the path of the keras model to load and play against
        :param num_sim: number of MCTS simulation
        :return: None, it's just a game, have fun !
        """
        # Initializing the game
        game = TicTacToe()

        # Initializing MCTS
        mcts = MCTS()

        # Loading the specified keras model
        model = keras.models.load_model(pathname)

        # Getting the first player to go and the initial empty tic tac toe board state
        state, player = game.reset()

        game.print_state(state)

        # If 1 human goes first else machine goes first
        first_to_go = np.random.randint(2)

        if first_to_go:
            human, ai = 1, -1
        else:
            human, ai = -1, 1

        while not game.is_terminal(state):
            if player == human:
                print("Your turn to choose, select a free slot on the board (integer between 0 and 9) with no piece on it")
                action = input()
                while not action.isdigit() or int(action) < 0 or int(action) > 8 or not game.is_valid(state, int(action)):
                    print("Invalid input, choose an integer between 0 and 9 and make sure to target a free cell")
                    action = input()
                action = int(action)
            else:
                print("AI's turn, one must wonder what it's thinking right now ...")
                for _ in range(num_sim):
                    mcts.search(state, game, player, model)
                pi = mcts.best_policy(state, game, tau=0.01)
                action = np.random.choice(game.action_space(), p=pi)
            state, player = game.step(state, action, player)
            game.print_state(state)

        if game.player_win(state, human):
            print("You beat AlphaZero ! Lee Sedol is very jealous right now ...")
        elif game.player_win(state, ai):
            print("You lost, the age of machines has now begun ...")
        else:
            print("Tie ! Human and machine are equal after all ... ?")


if __name__ == "__main__":

    play = Play()
    play.human_vs_ai("best_net")
