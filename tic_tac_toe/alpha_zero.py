"""
Module used for training an agent to play a game using self play with the Alpha Zero algorithm
"""
import numpy as np
import tensorflow as tf
from sklearn.model_selection import KFold
from mcts import MCTS
from nnet import TicTacToeNet
from tictactoe import TicTacToe
from tensorflow import keras
from play import Play


class AlphaZero:

    def __init__(self, game, num_iter=1000, num_games=100, num_sim=50, num_plays=100, tau_iter=30, lr=1e-3):
        """
        Constructor of the alpha zero instance which trains an agent for a specific game
        """
        self.game = game
        self.num_iter = num_iter
        self.num_games = num_games
        self.num_sim = num_sim
        self.num_plays = num_plays
        self.lr = lr
        self.tau_iter = tau_iter

    def train(self):
        """
        Function used to train our policy and value network through self play
        We go through num_iter iterations of training, for each iteration we self play num_games times.
        At the end of each game we update our neural network if it performs better than the preceding one with
        a win rate superior to 55%
        :return: the best version of the neural networks obtained through self play
        """
        # Initializing tic tac toe
        game = TicTacToe()

        # Initializing play
        play = Play()

        # Compiling the model for training
        best_net = TicTacToeNet()
        best_net.compile(
            optimizer=keras.optimizers.Adam(self.lr),
            loss=["categorical_crossentropy", "mean_squared_error"]
        )

        # Initializing batch that will contain training examples
        X, y_1, y_2 = [], [], []

        # Time to be patient
        for i in range(self.num_iter):
            for j in range(self.num_games):
                X_batch, y_1_batch, y_2_batch = self.self_play(game, best_net)
                X, y_1, y_2 = X + X_batch, y_1 + y_1_batch, y_2 + y_2_batch

            # In theory I would need to evaluate my nn against the previous ones but for now lets make things simple
            kf = KFold(n_splits=5)

            # Training current net with previous net using
            #best_net = TicTacToeNet()
            #best_net.compile(
            #    optimizer=keras.optimizers.Adam(self.lr),
            #    loss=["categorical_crossentropy", "mean_squared_error"]
            #)

            for train_idx, test_idx in kf.split(X):
                best_net.fit(
                    tf.convert_to_tensor(np.array(X)[train_idx]),
                    [tf.convert_to_tensor(np.array(y_1)[train_idx]), tf.convert_to_tensor(np.array(y_2)[train_idx])],
                    batch_size=32,
                    validation_data=(tf.convert_to_tensor(np.array(X)[test_idx]),
                                     [tf.convert_to_tensor(np.array(y_1)[test_idx]),
                                     tf.convert_to_tensor(np.array(y_2)[test_idx])])
                )

            #wins = []

            #for _ in range(self.num_plays):
            #    wins.append(play.ai_vs_ai(curr_net, best_net, verbose=False))
            #win_rate = sum(wins) / self.num_plays
            #if win_rate >= 0.55:
            #    curr_net.save("best_net")
            #    print("Iteration", i, "a new neural net prevailed ...")

            #if i == 0:
            #    best_net.save("best_net")

            best_net.save("best_net")
            #best_net = keras.models.load_model("best_net")

        return best_net

    def self_play(self, game, nnet):
        """
        A round of self play. Until a winner is decided, keep playing the game. Decisions are made thanks
        to a MCTS for a fixed number of simulations for each play. A list of training examples are generated
        :param game: the game that is being played
        :param nnet: the neural net used to make predictions of pi and v
        :return: X, y arrays of training examples used to train the neural network
        """
        # Initializing the game
        state, player = game.reset()

        # Initializing monte carlo tree search
        mcts = MCTS()

        # Initializing arrays that will contain trianing examples
        X, y_1, y_2 = [], [], []

        # Initializing tau, will be set to an infinitesimal value after 29 iterations
        taus = [1.0 if i < self.tau_iter else 0.01 for i in range(self.num_sim)]

        # Initializing counter for taus
        i = 0

        while True:

            # Launch num_sim MCTS simulatons
            for sim in range(self.num_sim):
                mcts.search(state, game, player, nnet)

            # Computing improved policy
            pi = mcts.best_policy(state, game, taus[i])

            # Add the state and associated policy improved by MCTS
            X.append(state.reshape(3, 3, 1))
            y_1.append(pi)

            # Deciding next action based on improved policy
            action = np.random.choice(game.action_space(), p=pi)
            state, player = game.step(state, action, player)

            # Changing temperature
            i += 1

            #print(state)

            if game.is_terminal(state):
                # Since player 1 starts check if he won and alternatively negate reward to be added to training examples
                reward = game.reward(state, 1)

                for i in range(len(y_1)):
                    y_2.append(reward * ((-1) ** i))

                return X, y_1, y_2
