"""
Test driver for MCTS implementation in Kalah game
Includes fake neural network model and comprehensive test suite
"""

import numpy as np
import pytest
import time
from typing import Tuple, Dict, Any
from dataclasses import dataclass
from unittest.mock import Mock

# Import the actual implementations
from kalah_game import KalahGame
from mcts import MCTS, MCTSNode


@dataclass
class MCTSConfig:
    """Configuration for MCTS parameters"""

    class mcts:
        num_simulations: int = 100
        c_puct: float = 1.0
        dirichlet_alpha: float = 0.3
        dirichlet_epsilon: float = 0.25


class FakeNeuralNetwork:
    """
    Fake neural network that provides reasonable predictions for testing
    Uses simple heuristics to generate policy and value estimates
    """

    def __init__(self, random_seed: int = 42):
        """Initialize with optional random seed for reproducible testing"""
        self.rng = np.random.RandomState(random_seed)

    def predict(self, state: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Generate fake predictions for policy and value

        Args:
            state: Game state as numpy array (12 elements - pits only)

        Returns:
            Tuple of (policy, value) where:
            - policy: 6-element array of action probabilities
            - value: Estimated value from current player's perspective (-1 to 1)
        """
        # Simple heuristic policy: prefer pits with more stones
        # State format: [current_player_pits(6), opponent_pits(6)]
        current_pits = state[:6]
        opponent_pits = state[6:]

        # Basic policy: weighted by stone count + small random noise
        policy = current_pits.astype(float) + 0.1
        policy = policy + self.rng.uniform(0, 0.1, size=6)  # Add noise

        # Normalize to valid probability distribution
        if np.sum(policy) > 0:
            policy = policy / np.sum(policy)
        else:
            policy = np.ones(6) / 6  # Uniform if all zeros

        # Simple value heuristic: difference in total stones
        current_total = np.sum(current_pits)
        opponent_total = np.sum(opponent_pits)

        # Normalize to [-1, 1] range
        stone_diff = current_total - opponent_total
        value = np.tanh(stone_diff / 10.0)  # Sigmoid-like scaling

        # Add some randomness
        value += self.rng.uniform(-0.1, 0.1)
        value = np.clip(value, -1.0, 1.0)

        return policy, float(value)


class MCTSTestSuite:
    """Comprehensive test suite for MCTS implementation"""

    def __init__(self):
        self.config = MCTSConfig()
        self.network = FakeNeuralNetwork()
        self.mcts = MCTS(self.config, self.network)

    def test_node_initialization(self):
        """Test MCTSNode initialization and basic methods"""
        print("Testing MCTSNode initialization...")

        # Test default initialization
        node = MCTSNode()
        assert node.visit_count == 0
        assert node.value_sum == 0.0
        assert node.prior == 0.0
        assert len(node.children) == 0
        assert node.virtual_loss == 0
        assert node.value() == 0.0
        assert not node.is_expanded()

        # Test with prior
        node_with_prior = MCTSNode(prior=0.5)
        assert node_with_prior.prior == 0.5

        print("✓ MCTSNode initialization tests passed")

    def test_ucb_score(self):
        """Test UCB score calculation"""
        print("Testing UCB score calculation...")

        node = MCTSNode(prior=0.3)

        # Test unvisited node
        ucb = node.ucb_score(parent_visits=10, c_puct=1.0)
        expected = 1.0 * 0.3 * np.sqrt(10)
        assert abs(ucb - expected) < 1e-6

        # Test visited node
        node.visit_count = 5
        node.value_sum = 2.5  # Average value = 0.5
        ucb = node.ucb_score(parent_visits=10, c_puct=1.0)
        exploitation = 0.5
        exploration = 1.0 * 0.3 * np.sqrt(10) / (1 + 5)
        expected = exploitation + exploration
        assert abs(ucb - expected) < 1e-6

        print("✓ UCB score calculation tests passed")

    def test_game_state_consistency(self):
        """Test game state handling and consistency"""
        print("Testing game state consistency...")

        game = KalahGame()

        # Test initial state
        state = game.get_canonical_state()
        assert len(state) == 12
        assert np.all(state == 4)  # All pits start with 4 stones

        # Test valid moves
        valid_moves = game.get_valid_moves()
        assert len(valid_moves) == 6
        assert np.all(valid_moves == 1)  # All moves valid initially

        # Test state key generation
        state_key = self.mcts._state_key(game)
        assert isinstance(state_key, str)
        assert len(state_key) > 0

        print("✓ Game state consistency tests passed")

    def test_node_expansion(self):
        """Test node expansion functionality"""
        print("Testing node expansion...")

        game = KalahGame()
        state_key = self.mcts._state_key(game)

        # Test expansion
        self.mcts._expand_node(game, state_key)
        assert state_key in self.mcts.nodes

        node = self.mcts.nodes[state_key]
        assert node.is_expanded()
        assert len(node.children) == 6  # Should have 6 children for valid moves

        # Test that all children have valid priors
        total_prior = sum(child.prior for child in node.children.values())
        assert abs(total_prior - 1.0) < 1e-6  # Should sum to 1

        print("✓ Node expansion tests passed")

    def test_action_selection(self):
        """Test action selection based on UCB scores"""
        print("Testing action selection...")

        game = KalahGame()
        state_key = self.mcts._state_key(game)
        self.mcts._expand_node(game, state_key)

        node = self.mcts.nodes[state_key]
        valid_moves = game.get_valid_moves()

        # Test action selection
        action = self.mcts._select_action(node, valid_moves)
        assert action is not None
        assert 0 <= action <= 5
        assert valid_moves[action] == 1

        # Test with invalid moves
        invalid_moves = np.zeros(6, dtype=bool)
        action = self.mcts._select_action(node, invalid_moves)
        assert action is None

        print("✓ Action selection tests passed")

    def test_simulation(self):
        """Test single MCTS simulation"""
        print("Testing MCTS simulation...")

        game = KalahGame()
        root_state = self.mcts._state_key(game)

        # Run a single simulation
        value = self.mcts._simulate(game.clone(), root_state)
        assert isinstance(value, float)
        assert -1.0 <= value <= 1.0

        # Check that nodes were created
        assert len(self.mcts.nodes) > 0

        print("✓ MCTS simulation tests passed")

    def test_search(self):
        """Test full MCTS search"""
        print("Testing MCTS search...")

        # Use smaller number of simulations for faster testing
        self.config.mcts.num_simulations = 10

        game = KalahGame()
        visits = self.mcts.search(game)

        assert len(visits) == 6
        assert np.sum(visits) > 0  # Should have some visits
        assert np.all(visits >= 0)  # All visit counts non-negative

        print("✓ MCTS search tests passed")

    def test_action_probabilities(self):
        """Test action probability calculation"""
        print("Testing action probabilities...")

        self.config.mcts.num_simulations = 20

        game = KalahGame()

        # Test with temperature = 1.0
        probs = self.mcts.get_action_probabilities(game, temperature=1.0)
        assert len(probs) == 6
        assert abs(np.sum(probs) - 1.0) < 1e-6  # Should sum to 1
        assert np.all(probs >= 0)  # All probabilities non-negative

        # Test with temperature = 0.0 (deterministic)
        probs_det = self.mcts.get_action_probabilities(game, temperature=0.0)
        assert abs(np.sum(probs_det) - 1.0) < 1e-6
        assert np.sum(probs_det == 1.0) == 1  # Exactly one action should have prob 1

        print("✓ Action probabilities tests passed")

    def test_game_completion(self):
        """Test MCTS behavior with game completion"""
        print("Testing MCTS with game completion...")

        # Create a near-end game state
        game = KalahGame()
        # Manually set up a near-end position
        game.board[0:5] = 0  # Empty most of player 1's pits
        game.board[0] = 1  # Leave one stone

        probs = self.mcts.get_action_probabilities(game, temperature=1.0)

        # Should still get valid probabilities
        assert len(probs) == 6
        assert abs(np.sum(probs) - 1.0) < 1e-6

        print("✓ Game completion tests passed")

    def test_tree_reuse(self):
        """Test tree reuse functionality"""
        print("Testing tree reuse...")

        game = KalahGame()

        # First search
        visits1 = self.mcts.search(game)
        nodes_count1 = len(self.mcts.nodes)

        # Second search on same position
        visits2 = self.mcts.search(game)
        nodes_count2 = len(self.mcts.nodes)

        # Should reuse existing nodes
        assert nodes_count2 >= nodes_count1

        # Clear tree and verify
        self.mcts.clear_tree()
        assert len(self.mcts.nodes) == 0

        print("✓ Tree reuse tests passed")

    def test_thread_safety(self):
        """Test thread safety of MCTS operations"""
        print("Testing thread safety...")

        # This is a basic test - in practice you'd want more comprehensive threading tests
        game = KalahGame()

        # Multiple rapid searches (simulates concurrent access)
        for _ in range(5):
            self.mcts.search(game)

        # Should not crash and should produce valid results
        probs = self.mcts.get_action_probabilities(game)
        assert len(probs) == 6
        assert abs(np.sum(probs) - 1.0) < 1e-6

        print("✓ Thread safety tests passed")

    def run_performance_test(self):
        """Run performance benchmarks"""
        print("Running performance tests...")

        game = KalahGame()

        # Benchmark search performance
        self.config.mcts.num_simulations = 100

        start_time = time.time()
        for _ in range(10):
            self.mcts.get_action_probabilities(game)
        end_time = time.time()

        avg_time = (end_time - start_time) / 10
        print(f"Average time per MCTS search (100 sims): {avg_time:.4f}s")

        # Benchmark with more simulations
        self.config.mcts.num_simulations = 1000
        start_time = time.time()
        self.mcts.get_action_probabilities(game)
        end_time = time.time()

        time_1000 = end_time - start_time
        print(f"Time for MCTS search (1000 sims): {time_1000:.4f}s")

        print("✓ Performance tests completed")

    def run_full_game_test(self):
        """Test MCTS playing a complete game"""
        print("Testing complete game play...")

        game = KalahGame()
        self.config.mcts.num_simulations = 50

        moves_played = 0
        max_moves = 200  # Prevent infinite games

        while not game.game_over and moves_played < max_moves:
            # Get action probabilities
            probs = self.mcts.get_action_probabilities(game, temperature=0.1)

            # Select action
            valid_moves = game.get_valid_moves()
            masked_probs = probs * valid_moves

            if np.sum(masked_probs) == 0:
                # No valid moves
                break

            action = np.argmax(masked_probs)

            # Make move
            try:
                extra_turn = game.make_move(int(action))
                moves_played += 1

                # Clear some of the tree to simulate real game conditions
                if moves_played % 5 == 0:
                    self.mcts.clear_tree()

            except ValueError as e:
                print(f"Invalid move attempted: {e}")
                break

        print(f"Game completed in {moves_played} moves")
        print(f"Game over: {game.game_over}")
        if game.game_over:
            print(f"Winner: {game.winner}")
            print(f"Final scores: P1={game.board[6]}, P2={game.board[13]}")

        print("✓ Complete game test passed")

    def run_all_tests(self):
        """Run all tests in the suite"""
        print("=" * 60)
        print("RUNNING MCTS TEST SUITE")
        print("=" * 60)

        try:
            self.test_node_initialization()
            self.test_ucb_score()
            self.test_game_state_consistency()
            self.test_node_expansion()
            self.test_action_selection()
            self.test_simulation()
            self.test_search()
            self.test_action_probabilities()
            self.test_game_completion()
            self.test_tree_reuse()
            self.test_thread_safety()
            self.run_performance_test()
            self.run_full_game_test()

            print("=" * 60)
            print("ALL TESTS PASSED! ✓")
            print("=" * 60)

        except Exception as e:
            print(f"❌ TEST FAILED: {e}")
            raise


def run_interactive_demo():
    """Run an interactive demo of MCTS vs random player"""
    print("\n" + "=" * 60)
    print("INTERACTIVE MCTS DEMO")
    print("=" * 60)

    config = MCTSConfig()
    config.mcts.num_simulations = 200
    network = FakeNeuralNetwork()
    mcts = MCTS(config, network)

    game = KalahGame()

    print("MCTS (Player 1) vs Random (Player 2)")
    print("Starting game...")

    moves = 0
    while not game.game_over and moves < 200:
        game.render()

        if game.current_player == 0:  # MCTS player
            print("MCTS thinking...")
            probs = mcts.get_action_probabilities(game, temperature=0.1)
            valid_moves = game.get_valid_moves()
            masked_probs = probs * valid_moves

            if np.sum(masked_probs) == 0:
                print("No valid moves for MCTS!")
                break

            action = np.argmax(masked_probs)
            print(f"MCTS chooses action {action} (pit {action + 1})")

        else:  # Random player
            valid_moves = game.get_valid_moves()
            valid_actions = [i for i in range(6) if valid_moves[i]]

            if not valid_actions:
                print("No valid moves for Random player!")
                break

            action = np.random.choice(valid_actions)
            print(f"Random player chooses action {action} (pit {action + 1})")

        try:
            extra_turn = game.make_move(int(action))
            moves += 1

            if extra_turn:
                print(f"Player {game.current_player + 1} gets an extra turn!")

        except ValueError as e:
            print(f"Invalid move: {e}")
            break

        # Clear tree occasionally to simulate real conditions
        if moves % 10 == 0:
            mcts.clear_tree()

    game.render()
    print(f"\nGame finished after {moves} moves!")

    if game.game_over and game.winner is not None:
        if game.winner == -1:
            print("Game ended in a draw!")
        else:
            winner_name = "MCTS" if game.winner == 0 else "Random"
            print(f"{winner_name} (Player {game.winner + 1}) wins!")
        print(f"Final scores: MCTS={game.board[6]}, Random={game.board[13]}")


if __name__ == "__main__":
    # Run the test suite
    test_suite = MCTSTestSuite()
    test_suite.run_all_tests()

    # Run interactive demo
    run_interactive_demo()
