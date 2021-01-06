"""
Main module for learning tic tac toe through self play using Alpha Zero
"""
from alpha_zero import AlphaZero
from tictactoe import TicTacToe

# Necessary global variables for training
NUM_ITER = 1000
NUM_GAMES = 100
NUM_SIM = 50
NUM_PLAYS = 100
LEARNING_RATE = 1e-3
TAU_ITER = 5

if __name__ == "__main__":

    game = TicTacToe()

    az = AlphaZero(
        game,
        num_iter=NUM_ITER,
        num_games=NUM_GAMES,
        num_sim=NUM_SIM,
        num_plays=NUM_PLAYS,
        tau_iter=TAU_ITER,
        lr=LEARNING_RATE
    )

    nnet = az.train()

    print("Training successful")
