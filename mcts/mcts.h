#ifndef MCTS_H
#define MCTS_H

#include <vector>
#include <string>
#include <memory>
#include <cmath>
#include <limits>
#include <random>
#include <map>
#include <thread>
#include <mutex>
#include <atomic>
#include <future>

#include "../include/llama.h"
#include "../common/common.h"
#include "../common/sampling.h"

// Forward declarations from llama.cpp
struct llama_context;
struct llama_model;
struct llama_vocab;
struct common_sampler; // From common.h

namespace mcts {

struct GameState {
    std::vector<int> tokens_sequence; // llama_token is int32_t, using int for simplicity here.
    bool is_terminal = false;
    float reward = 0.0f; // Calculated at terminal state

    GameState(const std::vector<int>& initial_tokens = {});
    bool is_terminal_state(const struct llama_vocab* vocab, int max_length) const; // vocab for EOS
    // Reward calculation might be better placed in MCTS or MCTSNode after simulation
};

class MCTSNode : public std::enable_shared_from_this<MCTSNode> {
public:
    std::shared_ptr<MCTSNode> parent;
    GameState state;
    std::vector<int> untried_actions;
    std::vector<std::shared_ptr<MCTSNode>> children;

    std::atomic<double> q_value{0.0};
    std::atomic<int> n_visits{0};
    mutable std::mutex node_mutex; // For thread-safe operations on children and untried_actions

    // Pointers to llama context for node operations (expansion, simulation trigger)
    struct llama_context* l_ctx;
    struct llama_model* l_model;
    const struct llama_vocab* l_vocab;
    struct common_sampler* l_sampler; // Sampler to use for getting actions/probabilities

    MCTSNode(GameState s,
             struct llama_context* ctx, struct llama_model* model, const struct llama_vocab* vocab, struct common_sampler* sampler,
             std::shared_ptr<MCTSNode> p = nullptr);

    bool is_fully_expanded() const;
    bool is_terminal_node() const;
    std::shared_ptr<MCTSNode> uct_select_child(double exploration_constant) const;
    std::shared_ptr<MCTSNode> expand(); // Will use l_ctx, l_model, etc.
    void backpropagate(double reward_value);

private:
    void populate_untried_actions(); // Helper to get actions from l_ctx
};

class MCTS {
public:
    MCTS(struct llama_context* ctx, struct llama_model* model, const struct llama_vocab* vocab, struct common_sampler* sampler,
         double exploration_constant = 1.414);

    void run_iteration(std::shared_ptr<MCTSNode> root_node);
    void run_parallel_iterations(std::shared_ptr<MCTSNode> root_node, int num_iterations, int num_threads);
    int get_best_action(std::shared_ptr<MCTSNode> root_node, int num_iterations); // Returns action (token ID)

private:
    std::shared_ptr<MCTSNode> select_promising_node(std::shared_ptr<MCTSNode> root_node);
    // expand_node is now part of MCTSNode::expand
    GameState simulate_playout(std::shared_ptr<MCTSNode> node); // Renamed, more descriptive
    double simulate_lightweight(std::shared_ptr<MCTSNode> node); // Fast heuristic-based simulation
    void backpropagate_rewards(std::shared_ptr<MCTSNode> node, double reward_value);

    struct llama_context* l_ctx_;
    struct llama_model* l_model_;
    const struct llama_vocab* l_vocab_;
    struct common_sampler* l_sampler_; // Sampler passed from server_slot

    double exploration_constant_;
    std::mt19937 random_engine_;
    mutable std::mutex mcts_mutex_; // For thread-safe operations
};

} // namespace mcts

#endif // MCTS_H