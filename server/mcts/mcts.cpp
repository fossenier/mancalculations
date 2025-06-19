#include "mcts.hpp"
#include "kalah_game.hpp"
#include "neural_network.hpp"
#include <algorithm>
#include <numeric>
#include <sstream>
#include <iomanip>

// Thread-local RNG
thread_local std::mt19937 MCTS::rng_{std::random_device{}()};

// MCTSNode implementation
MCTSNode::MCTSNode(float prior) : prior(prior) {}

float MCTSNode::value() const
{
    int visits = visit_count.load(std::memory_order_acquire);
    if (visits == 0)
        return 0.0f;
    return value_sum.load(std::memory_order_acquire) / visits;
}

float MCTSNode::ucb_score(int parent_visits, float c_puct) const
{
    int visits = visit_count.load(std::memory_order_acquire);

    if (visits == 0)
    {
        return c_puct * prior * std::sqrt(static_cast<float>(parent_visits));
    }

    float exploitation = value();
    float exploration = c_puct * prior * std::sqrt(static_cast<float>(parent_visits)) / (1 + visits);
    float virtual_penalty = virtual_loss.load(std::memory_order_acquire) * c_puct / (1 + visits);

    return exploitation + exploration - virtual_penalty;
}

bool MCTSNode::is_expanded() const
{
    return expanded.load(std::memory_order_acquire);
}

bool MCTSNode::try_expand()
{
    bool expected = false;
    return expanded.compare_exchange_strong(expected, true, std::memory_order_acq_rel);
}

void MCTSNode::add_visit(float value)
{
    visit_count.fetch_add(1, std::memory_order_acq_rel);
    // Atomic float addition
    float old_sum = value_sum.load(std::memory_order_acquire);
    while (!value_sum.compare_exchange_weak(old_sum, old_sum + value,
                                            std::memory_order_acq_rel,
                                            std::memory_order_acquire))
        ;
}

void MCTSNode::add_virtual_loss()
{
    virtual_loss.fetch_add(1, std::memory_order_acq_rel);
}

void MCTSNode::remove_virtual_loss()
{
    virtual_loss.fetch_sub(1, std::memory_order_acq_rel);
}

// GPUBatchProcessor implementation
GPUBatchProcessor::GPUBatchProcessor(int gpu_id, int batch_size)
    : gpu_id_(gpu_id), batch_size_(batch_size)
{
    network_ = std::make_unique<NeuralNetwork>(gpu_id);
}

GPUBatchProcessor::~GPUBatchProcessor()
{
    stop();
}

void GPUBatchProcessor::start()
{
    running_ = true;
    worker_thread_ = std::thread(&GPUBatchProcessor::process_batches, this);
}

void GPUBatchProcessor::stop()
{
    running_ = false;
    cv_.notify_all();
    if (worker_thread_.joinable())
    {
        worker_thread_.join();
    }
}

void GPUBatchProcessor::submit_request(InferenceRequest &&request)
{
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        request_queue_.push(std::move(request));
    }
    cv_.notify_one();
}

void GPUBatchProcessor::process_batches()
{
    std::vector<InferenceRequest> batch;
    batch.reserve(batch_size_);

    while (running_)
    {
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            cv_.wait(lock, [this]
                     { return !request_queue_.empty() || !running_; });

            if (!running_)
                break;

            // Collect batch
            while (!request_queue_.empty() && batch.size() < batch_size_)
            {
                batch.push_back(std::move(request_queue_.front()));
                request_queue_.pop();
            }
        }

        if (!batch.empty())
        {
            // Prepare batch data
            std::vector<float> batch_states;
            batch_states.reserve(batch.size() * batch[0].state.size());

            for (const auto &req : batch)
            {
                batch_states.insert(batch_states.end(), req.state.begin(), req.state.end());
            }

            // Run inference
            auto [policies, values] = network_->predict_batch(batch_states, batch.size());

            // Distribute results
            for (size_t i = 0; i < batch.size(); ++i)
            {
                std::vector<float> policy(policies.begin() + i * 6, policies.begin() + (i + 1) * 6);
                batch[i].promise.set_value({policy, values[i]});
            }

            batch.clear();
        }
    }
}

// MCTS implementation
MCTS::MCTS(const MCTSConfig &config, int num_threads, int num_gpus)
    : config_(config), num_threads_(num_threads)
{

    // Initialize GPU processors
    for (int i = 0; i < num_gpus; ++i)
    {
        gpu_processors_.push_back(std::make_unique<GPUBatchProcessor>(i, 64));
        gpu_processors_[i]->start();
    }

    // Initialize thread pool
    for (int i = 0; i < num_threads_; ++i)
    {
        thread_pool_.emplace_back([this]
                                  {
            while (!stop_threads_) {
                std::function<void()> task;
                {
                    std::unique_lock<std::mutex> lock(task_mutex_);
                    task_cv_.wait(lock, [this] { return !task_queue_.empty() || stop_threads_; });
                    
                    if (stop_threads_) break;
                    
                    task = std::move(task_queue_.front());
                    task_queue_.pop();
                }
                task();
            } });
    }
}

MCTS::~MCTS()
{
    // Stop thread pool
    stop_threads_ = true;
    task_cv_.notify_all();
    for (auto &thread : thread_pool_)
    {
        thread.join();
    }

    // GPU processors are automatically stopped in their destructors
}

std::string MCTS::state_key(const KalahGame &game) const
{
    std::stringstream ss;
    ss << game.get_current_player() << ":";

    auto state = game.get_canonical_state();
    for (float val : state)
    {
        ss << std::setprecision(0) << std::fixed << val << ",";
    }

    return ss.str();
}

std::shared_ptr<MCTSNode> MCTS::get_or_create_node(const std::string &key)
{
    // Try to find existing node without lock
    auto it = nodes_.find(key);
    if (it != nodes_.end())
    {
        return it->second;
    }

    // Need to create - use lock
    std::lock_guard<std::mutex> lock(nodes_mutex_);

    // Double-check after acquiring lock
    it = nodes_.find(key);
    if (it != nodes_.end())
    {
        return it->second;
    }

    // Create new node
    auto node = std::make_shared<MCTSNode>();
    nodes_[key] = node;
    return node;
}

std::vector<float> MCTS::search(const KalahGame &game, const std::string &root_state_key)
{
    std::string root_key = root_state_key.empty() ? state_key(game) : root_state_key;

    // Initialize root node
    auto root_node = get_or_create_node(root_key);
    root_node->current_player = game.get_current_player();

    // Expand root if needed
    if (!root_node->is_expanded())
    {
        auto game_copy = game.clone();
        auto state = game_copy.get_canonical_state();
        auto [policy, _] = predict(state);
        expand_node(game_copy, root_key, policy);
    }

    // Add Dirichlet noise to root
    auto valid_moves = game.get_valid_moves();
    std::gamma_distribution<float> gamma(config_.dirichlet_alpha, 1.0f);
    std::vector<float> noise(6);
    float noise_sum = 0.0f;

    for (int i = 0; i < 6; ++i)
    {
        if (valid_moves[i])
        {
            noise[i] = gamma(rng_);
            noise_sum += noise[i];
        }
    }

    // Normalize noise and apply to root children
    for (int action = 0; action < 6; ++action)
    {
        if (valid_moves[action] && root_node->children.count(action))
        {
            auto child = root_node->children[action];
            child->prior = (1 - config_.dirichlet_epsilon) * child->prior +
                           config_.dirichlet_epsilon * (noise[action] / noise_sum);
        }
    }

    // Run simulations in batches
    int simulations_done = 0;
    while (simulations_done < config_.num_simulations)
    {
        int batch_size = std::min(64, config_.num_simulations - simulations_done);
        simulate_batch(batch_size, game, root_key);
        simulations_done += batch_size;
    }

    // Extract visit counts
    std::vector<float> visits(6, 0.0f);
    for (const auto &[action, child] : root_node->children)
    {
        visits[action] = static_cast<float>(child->visit_count.load(std::memory_order_acquire));
    }

    return visits;
}

void MCTS::simulate_batch(int batch_size, const KalahGame &root_game, const std::string &root_state)
{
    std::vector<std::future<void>> futures;

    for (int i = 0; i < batch_size; ++i)
    {
        auto promise = std::make_shared<std::promise<void>>();
        futures.push_back(promise->get_future());

        // Submit task to thread pool
        {
            std::lock_guard<std::mutex> lock(task_mutex_);
            task_queue_.push([this, root_game, root_state, promise]()
                             {
                auto task = simulate_to_leaf(root_game, root_state);
                
                if (task.game->is_game_over()) {
                    // Terminal node - backup immediately
                    float value = task.game->get_reward(task.game->get_current_player());
                    backup_lockfree(task.path, value);
                } else {
                    // Need neural network evaluation
                    auto state = task.game->get_canonical_state();
                    InferenceRequest request;
                    request.state = state;
                    
                    auto result_future = request.promise.get_future();
                    
                    // Submit to GPU
                    int gpu_id = next_gpu_.fetch_add(1) % gpu_processors_.size();
                    gpu_processors_[gpu_id]->submit_request(std::move(request));
                    
                    // Wait for result
                    auto [policy, value] = result_future.get();
                    
                    // Expand if needed
                    auto node = get_or_create_node(state_key(*task.game));
                    if (node->try_expand()) {
                        expand_node(*task.game, state_key(*task.game), policy);
                    }
                    
                    // Backup
                    backup_lockfree(task.path, value);
                }
                
                promise->set_value(); });
        }
        task_cv_.notify_one();
    }

    // Wait for all simulations to complete
    for (auto &future : futures)
    {
        future.wait();
    }
}

MCTS::SimulationTask MCTS::simulate_to_leaf(const KalahGame &game, const std::string &root_state)
{
    SimulationTask task;
    task.game = std::make_unique<KalahGame>(game);
    task.root_state = root_state;

    std::string current_state = root_state;

    while (true)
    {
        auto node = get_or_create_node(current_state);
        node->current_player = task.game->get_current_player();

        // Check terminal
        if (task.game->is_game_over())
        {
            return task;
        }

        // Check if needs expansion
        if (!node->is_expanded())
        {
            return task;
        }

        auto valid_moves = task.game->get_valid_moves();
        bool has_valid_moves = std::any_of(valid_moves.begin(), valid_moves.end(),
                                           [](bool v)
                                           { return v; });
        if (!has_valid_moves)
        {
            return task;
        }

        // Select action
        int action = select_action_lockfree(*node, valid_moves);
        if (action == -1)
        {
            return task;
        }

        // Add virtual loss
        auto child = node->children[action];
        child->add_virtual_loss();

        task.path.push_back({current_state, action, task.game->get_current_player()});

        // Make move
        bool extra_turn = task.game->make_move(action);
        current_state = state_key(*task.game);

        if (!extra_turn && task.game->is_game_over())
        {
            return task;
        }
    }
}

int MCTS::select_action_lockfree(const MCTSNode &node, const std::vector<bool> &valid_moves)
{
    int parent_visits = std::max(1, node.visit_count.load(std::memory_order_acquire));

    float best_score = -std::numeric_limits<float>::infinity();
    int best_action = -1;

    for (int action = 0; action < 6; ++action)
    {
        if (!valid_moves[action] || node.children.find(action) == node.children.end())
        {
            continue;
        }

        float score = node.children.at(action)->ucb_score(parent_visits, config_.c_puct);

        if (score > best_score)
        {
            best_score = score;
            best_action = action;
        }
    }

    return best_action;
}

void MCTS::expand_node(KalahGame &game, const std::string &state_key, const std::vector<float> &policy)
{
    auto node = nodes_[state_key];

    // Mask invalid actions
    auto valid_moves = game.get_valid_moves();
    std::vector<float> masked_policy(6);
    float policy_sum = 0.0f;

    for (int i = 0; i < 6; ++i)
    {
        if (valid_moves[i])
        {
            masked_policy[i] = policy[i];
            policy_sum += policy[i];
        }
    }

    // Normalize
    if (policy_sum > 0)
    {
        for (float &p : masked_policy)
        {
            p /= policy_sum;
        }
    }
    else
    {
        // Uniform over valid moves
        int valid_count = std::count(valid_moves.begin(), valid_moves.end(), true);
        for (int i = 0; i < 6; ++i)
        {
            if (valid_moves[i])
            {
                masked_policy[i] = 1.0f / valid_count;
            }
        }
    }

    // Create children
    for (int action = 0; action < 6; ++action)
    {
        if (valid_moves[action])
        {
            node->children[action] = std::make_shared<MCTSNode>(masked_policy[action]);
        }
    }
}

void MCTS::backup_lockfree(const std::vector<std::tuple<std::string, int, int>> &path, float value)
{
    float current_value = value;

    // Process path in reverse
    for (auto it = path.rbegin(); it != path.rend(); ++it)
    {
        const auto &[state_key, action, player] = *it;

        auto node = nodes_[state_key];
        auto child = node->children[action];

        // Remove virtual loss and add visit
        child->remove_virtual_loss();
        child->add_visit(current_value);
        node->add_visit(0); // Just increment visit count

        // Flip value for opponent
        current_value = -current_value;
    }
}

std::pair<std::vector<float>, float> MCTS::predict(const std::vector<float> &state)
{
    InferenceRequest request;
    request.state = state;

    auto future = request.promise.get_future();

    // Submit to next GPU in round-robin
    int gpu_id = next_gpu_.fetch_add(1) % gpu_processors_.size();
    gpu_processors_[gpu_id]->submit_request(std::move(request));

    return future.get();
}

void MCTS::clear_tree()
{
    std::lock_guard<std::mutex> lock(nodes_mutex_);
    nodes_.clear();
}

std::vector<float> MCTS::get_action_probabilities(const KalahGame &game, float temperature)
{
    auto visits = search(game);

    // Add small constant
    for (float &v : visits)
    {
        v += 1e-8f;
    }

    if (temperature < 1e-8f)
    {
        // Greedy selection
        std::vector<float> probs(6, 0.0f);
        int best_action = std::distance(visits.begin(), std::max_element(visits.begin(), visits.end()));
        probs[best_action] = 1.0f;
        return probs;
    }
    else
    {
        // Apply temperature
        std::vector<float> visits_temp(6);
        float sum = 0.0f;

        for (int i = 0; i < 6; ++i)
        {
            visits_temp[i] = std::pow(visits[i], 1.0f / temperature);
            sum += visits_temp[i];
        }

        // Normalize
        for (float &v : visits_temp)
        {
            v /= sum;
        }

        return visits_temp;
    }
}