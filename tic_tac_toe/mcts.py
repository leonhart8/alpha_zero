"""
Module which defines the class performing Monte Carlo Tree Search.
"""
import numpy as np

class MCTS:
    """
    Class which defines the Monte Carlo Tree Search. Given a state from a tic tac toe game we :
    - Add the state to the tree if it is not in the tree yet
    - Recursively call MCTS on a state if it is already in the tree
    - If a state is terminal, return the reward and let recursion perform back-up
    """

    def __init__(self, c=1):
        """
        Initializes the MCTS and its different tables :
        - Q which stores state action values
        - N which stores the amount of times each edge of the MCTS has been visited
        - P which stores the prior probabilities of selecting each action
        - nodes, which stores states that have been visited
        """
        self.Q, self.N, self.P, self.totals = {}, {}, {}, {}
        self.nodes = []
        self.c = c

    def search(self, state, game, player, neural_network):
        """
        Performs the famous Monte Carlo Tree Search.
        Given a state from a tic tac toe game we :
            - Add the state to the tree if it is not in the tree yet
            - Recursively call MCTS on a state if it is already in the tree
            - If a state is terminal, return the reward and let recursion perform back-up
        The backup values are negated because states alternate between players. As such a state
        that is winning for player 1 is losing for player 2
        :param state: the current state
        :param game: the game that we are playing
        :param player: the current player for a given state
        :param neural_network: the neural network used to compute prior probabilities and state values
        :return: the value associated to the last visited node for recursive backup
        """
        # If the game is over, backup the reward
        if game.is_terminal(state):
            return - game.reward(state, player)

        # If the state is not in the tree add it and backup its value
        elif game.hash_state(state) not in self.nodes:

            # Add state to tree nodes
            self.nodes.append(game.hash_state(state))

            # Computing hash of current state
            hash_state = game.hash_state(state)

            # Feed forward of neural network to get probabilities and values
            probas, v = neural_network(state.reshape(1, 3, 3, 1))
            probas, v = probas[0], v[0]

            # Masking actions that are not possible to take
            valid_actions = game.valid_actions(state)
            masked_probas = probas * valid_actions
            self.P[hash_state] = masked_probas

            # Normalizing new masked probabilities
            s = np.sum(masked_probas)
            if s > 0:
                masked_probas /= s
                self.P[hash_state] = masked_probas
            else:
                print("Weird, legal moves all set to 0 ?")

            # The total visits to a state
            self.totals[hash_state] = 0

            # Initializing the edges coming from the node
            self.N[hash_state] = np.zeros(game.action_space())

            # Initializing Q function
            self.Q[hash_state] = np.zeros(game.action_space())

            # Returning the opposite of the value for backup
            return -v

        # Getting the hash associated to the current state
        hash_state = game.hash_state(state)

        # Getting valid actions
        valid_actions = game.valid_actions(state)

        if not np.any(valid_actions):
            print("No valid actions but game not over ? Something's wrong")
            return

        # Now computing upper confidence bounds according to the paper
        best_u, best_a = -np.inf, -1
        for action in range(game.action_space()):
            if valid_actions[action]:
                usa = self.Q[hash_state][action] + self.c * self.P[hash_state][action] * (np.sqrt(self.totals[hash_state]) / (1 + self.N[hash_state][action]))
                if usa > best_u:
                    best_u, best_a = usa, action

        next_state, next_player = game.step(state, best_a, player)
        v = self.search(next_state, game, next_player, neural_network)

        # Online update of the value using the formula of mean online update
        self.Q[hash_state][best_a] = (self.N[hash_state][best_a] * self.Q[hash_state][best_a] + v) / (1 + self.N[hash_state][best_a])

        # Updating the edge visits
        self.N[hash_state][best_a] += 1

        # Updating the total number of visits
        self.totals[hash_state] += 1

        return -v

    def best_policy(self, state, game, tau=1):
        """
        This function computes the policy improved by MCTS
        :return: array of size equal to the game's action space with probabilities of picking each action
        """
        # Computing the hash code of the current state
        hash_state = game.hash_state(state)

        # array of all the counts of each action after the search, indeed the probability of picking an action
        # is proportional to its visit count
        counts = np.array([self.N[hash_state][action] ** (1/tau) for action in range(game.action_space())])

        # normalize the counts to get probability vector
        counts /= np.sum(counts)

        return counts
