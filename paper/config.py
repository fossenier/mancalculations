"""
Hyperparameters for my AlphaZero in Mancala implementation.
"""


# TODO adjust for Mancala
class AlphaZeroConfig(object):
    def __init__(self):
        ### Self-Play
        # Number of games played in parallel.
        # self.num_actors = 5000 # from paper
        self.num_cores = 60
        self.games_per_batch = 64
        self.game_batches_per_core = 4
        
        # Number of moves from start of game where the action is chosen
        # using stochastic policy
        self.num_sampling_moves = 30
        self.temperature = 1.0
        # Number of moves before forcing the game to end (Do I need this?)
        self.max_moves = 512  # for chess and shogi, 722 for Go.
        # Number of MCTS model calls in choosing each action
        self.num_simulations = 800

        # The root node of each MCTS will have the prior probabilities fuzzied
        # with Dirichlet noise
        # Alpha determines the spikiness (close to 1.0)  or uniformness (close to .0)
        # of the Dirichlet distribution.
        self.root_dirichlet_alpha = 0.3  # for chess, 0.03 for Go and 0.15 for shogi.
        # The % probability that Dirchlet noise represents the root node's children's priors
        self.root_exploration_fraction = 0.25

        # UCB formula
        self.pb_c_base = 19652
        self.pb_c_init = 1.25

        ### Training
        # Number of times the model will self train
        self.training_steps = int(700e3)
        # Number of self trainings before the model is saved
        self.checkpoint_interval = int(1e3)
        # Number of most recent games held in memory for training
        self.window_size = int(1e6)
        # Number of games to sample from the memory for each self training cycletraining
        self.batch_size = 4096

        # Pushes weights towards zero. Makes model more general, and helps it to
        # not just memorize the training data. AKA L2 regularization to prevent overfitting
        self.weight_decay = 1e-4
        # Gives "momentum" to the direction of model weight adjustment, learning in a
        # consistent direction rather than bouncing around
        self.momentum = 0.9
        # Controls how rapidly the model learns (i.e. how quickly the weights jump
        # around). Too high means it might get worse, too low means it might take
        # too long to converge.
        # Schedule for chess and shogi, Go starts at 2e-2 immediately.
        self.learning_rate_schedule = {0: 2e-1, 100e3: 2e-2, 300e3: 2e-3, 500e3: 2e-4}
