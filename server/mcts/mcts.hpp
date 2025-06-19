#pragma once

#include <vector>
#include <memory>
#include <atomic>
#include <unordered_map>
#include <string>
#include <mutex>
#include <cmath>
#include <random>
#include <thread>
#include <future>
#include <queue>

// Forward declarations
class KalahGame;
class NeuralNetwork;

// Configuration structure
struct MCTSConfig
{
    int num_simulations = 800;
    float c_puct = 1.25f;
    float dirichlet_alpha = 0.3f;
    float dirichlet_epsilon = 0.25f;
    int temperature_threshold = 30;
    float initial_temperature = 1.0f;
    float final_temperature = 0.0f;
};

// Lock-free MCTS Node
class MCTSNode
{
public:
    MCTSNode(float prior = 0.0f);

    // Atomic operations for lock-free access
    float value() const;
    float ucb_score(int parent_visits, float c_puct) const;
    bool is_expanded() const;
    bool try_expand();

    // Atomic updates
    void add_visit(float value);
    void add_virtual_loss();
    void remove_virtual_loss();

    // Public members (read-only after initialization)
    float prior;
    int current_player = -1;
    std::unordered_map<int, std::shared_ptr<MCTSNode>> children;

private:
    std::atomic<int> visit_count{0};
    std::atomic<float> value_sum{0.0f};
    std::atomic<int> virtual_loss{0};
    std::atomic<bool> expanded{false};
};

// Batch inference request
struct InferenceRequest
{
    std::vector<float> state;
    std::promise<std::pair<std::vector<float>, float>> promise;
};

// GPU Batch Processor
class GPUBatchProcessor
{
public:
    GPUBatchProcessor(int gpu_id, int batch_size);
    ~GPUBatchProcessor();

    void submit_request(InferenceRequest &&request);
    void start();
    void stop();

private:
    void process_batches();

    int gpu_id_;
    int batch_size_;
    std::queue<InferenceRequest> request_queue_;
    std::mutex queue_mutex_;
    std::condition_variable cv_;
    std::thread worker_thread_;
    std::atomic<bool> running_{false};
    std::unique_ptr<NeuralNetwork> network_;
};

// Main MCTS class
class MCTS
{
public:
    MCTS(const MCTSConfig &config, int num_threads = 64, int num_gpus = 4);
    ~MCTS();

    // Main search function
    std::vector<float> search(const KalahGame &game, const std::string &root_state = "");

    // Get action probabilities
    std::vector<float> get_action_probabilities(const KalahGame &game, float temperature = 1.0f);

    // Clear the search tree
    void clear_tree();

private:
    // Thread pool for parallel simulations
    struct SimulationTask
    {
        std::unique_ptr<KalahGame> game;
        std::string root_state;
        std::vector<std::tuple<std::string, int, int>> path;
        std::promise<void> done;
    };

    // Helper functions
    std::string state_key(const KalahGame &game) const;
    std::shared_ptr<MCTSNode> get_or_create_node(const std::string &key);

    // Simulation functions
    void simulate_batch(int batch_size, const KalahGame &root_game, const std::string &root_state);
    SimulationTask simulate_to_leaf(const KalahGame &game, const std::string &root_state);
    int select_action_lockfree(const MCTSNode &node, const std::vector<bool> &valid_moves);

    // Expansion and backup
    void expand_node(KalahGame &game, const std::string &state_key, const std::vector<float> &policy);
    void backup_lockfree(const std::vector<std::tuple<std::string, int, int>> &path, float value);

    // GPU inference
    std::pair<std::vector<float>, float> predict(const std::vector<float> &state);

    // Members
    MCTSConfig config_;
    std::unordered_map<std::string, std::shared_ptr<MCTSNode>> nodes_;
    std::mutex nodes_mutex_; // Only for node creation

    // Thread pool
    int num_threads_;
    std::vector<std::thread> thread_pool_;
    std::queue<std::function<void()>> task_queue_;
    std::mutex task_mutex_;
    std::condition_variable task_cv_;
    std::atomic<bool> stop_threads_{false};

    // GPU processors
    std::vector<std::unique_ptr<GPUBatchProcessor>> gpu_processors_;
    std::atomic<int> next_gpu_{0};

    // Random number generation
    thread_local static std::mt19937 rng_;
};
