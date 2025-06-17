"""
Test script to verify AlphaZero Kalah installation and basic functionality
"""

import torch
import numpy as np
import sys
import os
from datetime import datetime


def test_cuda():
    """Test CUDA availability and GPU count"""
    print("Testing CUDA...")
    if not torch.cuda.is_available():
        print("❌ CUDA is not available!")
        return False

    gpu_count = torch.cuda.device_count()
    print(f"✓ CUDA is available with {gpu_count} GPU(s)")

    for i in range(gpu_count):
        props = torch.cuda.get_device_properties(i)
        print(f"  GPU {i}: {props.name} ({props.total_memory / 1e9:.1f} GB)")

    if gpu_count < 4:
        print(f"⚠️  Warning: Found {gpu_count} GPUs, but 4 are recommended")

    return True


def test_imports():
    """Test all required imports"""
    print("\nTesting imports...")
    modules = [
        "config",
        "kalah_game",
        "network",
        "mcts",
        "trainer",
        "self_play",
        "evaluator",
        "monitor",
    ]

    success = True
    for module in modules:
        try:
            __import__(module)
            print(f"✓ {module}")
        except ImportError as e:
            print(f"❌ {module}: {e}")
            success = False

    return success


def test_game():
    """Test Kalah game implementation"""
    print("\nTesting Kalah game...")
    from kalah_game import KalahGame

    game = KalahGame()
    print("✓ Game initialization")

    # Test initial state
    assert np.sum(game.board) == 72, "Initial stone count should be 72"
    print("✓ Initial state correct")

    # Test move
    game.make_move(0)
    assert game.board[0] == 0, "Pit should be empty after move"
    print("✓ Move execution")

    # Test valid moves
    valid = game.get_valid_moves()
    assert len(valid) == 6, "Should have 6 move options"
    print("✓ Valid move detection")

    return True


def test_network():
    """Test neural network"""
    print("\nTesting neural network...")
    from config import get_config
    from network import KalahNetwork

    config = get_config()
    network = KalahNetwork(config)
    print("✓ Network initialization")

    # Test forward pass
    batch_size = 4
    dummy_input = torch.randn(batch_size, 14)
    log_policy, value = network(dummy_input)

    assert log_policy.shape == (
        batch_size,
        6,
    ), f"Policy shape mismatch: {log_policy.shape}"
    assert value.shape == (batch_size, 1), f"Value shape mismatch: {value.shape}"
    print("✓ Forward pass")

    # Test CUDA if available
    if torch.cuda.is_available():
        network = network.cuda()
        dummy_input = dummy_input.cuda()
        log_policy, value = network(dummy_input)
        print("✓ CUDA forward pass")

    return True


def test_mcts():
    """Test MCTS implementation"""
    print("\nTesting MCTS...")
    from config import get_config
    from network import KalahNetwork
    from mcts import MCTS
    from kalah_game import KalahGame

    config = get_config()
    config.mcts.num_simulations = 100  # Reduce for testing

    network = KalahNetwork(config)
    mcts = MCTS(config, network)
    game = KalahGame()

    visits = mcts.search(game)
    assert len(visits) == 6, "Should return 6 visit counts"
    assert np.sum(visits) > 0, "Should have non-zero visits"
    print("✓ MCTS search")

    return True


def test_distributed():
    """Test distributed training setup"""
    print("\nTesting distributed setup...")

    # Check environment variables
    required_env = ["MASTER_ADDR", "MASTER_PORT"]
    for env in required_env:
        if env not in os.environ:
            os.environ[env] = "localhost" if env == "MASTER_ADDR" else "12355"

    print("✓ Environment variables set")

    # Test process group initialization (single process)
    try:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()

        # Note: Full distributed test requires multiple processes
        print("✓ Distributed imports working")
        print(
            "  (Full distributed test requires launching with torch.distributed.launch)"
        )
    except Exception as e:
        print(f"⚠️  Distributed test limited: {e}")

    return True


def run_all_tests():
    """Run all tests"""
    print("=" * 60)
    print("AlphaZero Kalah Installation Test")
    print(f"Time: {datetime.now()}")
    print("=" * 60)

    tests = [
        ("CUDA", test_cuda),
        ("Imports", test_imports),
        ("Game", test_game),
        ("Network", test_network),
        ("MCTS", test_mcts),
        ("Distributed", test_distributed),
    ]

    results = []
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success))
        except Exception as e:
            print(f"❌ {name} test failed with error: {e}")
            results.append((name, False))

    print("\n" + "=" * 60)
    print("Test Summary:")
    print("=" * 60)

    all_passed = True
    for name, success in results:
        status = "✓ PASS" if success else "❌ FAIL"
        print(f"{name:<15} {status}")
        if not success:
            all_passed = False

    if all_passed:
        print("\n🎉 All tests passed! System is ready for training.")
    else:
        print("\n⚠️  Some tests failed. Please check the errors above.")
        sys.exit(1)


if __name__ == "__main__":
    run_all_tests()
