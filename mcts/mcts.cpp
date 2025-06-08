#include "mcts.h"
#include "llama.h"
#include "common.h" // For common_batch_add, common_sampler_*, common_token_to_piece, etc.
// sampling.h is included by common.h, provides llama_token_data_array definition.

#include <algorithm>
#include <vector>
#include <memory>
#include <random>
#include <cmath>
#include <limits>
#include <iostream> // For debugging and logging
#include <cstdint>  // For uint8_t

namespace mcts {

// --- GameState Implementation ---
GameState::GameState(const std::vector<int>& initial_tokens)
    : tokens_sequence(initial_tokens), is_terminal(false), reward(0.0f) {}

bool GameState::is_terminal_state(const struct llama_vocab* vocab, int max_length) const {
    if (is_terminal) return true;
    if (tokens_sequence.empty() && max_length != 0) return false; // Not terminal if empty unless max_length is 0 (no generation allowed)
    if (vocab && !tokens_sequence.empty() && tokens_sequence.back() == llama_vocab_eos(vocab)) return true;
    if (max_length > 0 && tokens_sequence.size() >= static_cast<size_t>(max_length)) return true;
    return false;
}

// --- MCTSNode Implementation ---
MCTSNode::MCTSNode(GameState s,
                   struct llama_context* ctx, struct llama_model* model, const struct llama_vocab* vocab, struct common_sampler* sampler,
                   std::shared_ptr<MCTSNode> p)
    : parent(p), state(s), l_ctx(ctx), l_model(model), l_vocab(vocab), l_sampler(sampler), n_visits(0), q_value(0.0) {
    populate_untried_actions();
}

void MCTSNode::populate_untried_actions() {
    untried_actions.clear();
    if (!l_ctx || !l_model || !l_vocab || state.is_terminal_state(l_vocab, llama_n_ctx(l_ctx))) {
        std::cout << "[MCTS] populate_untried_actions: Terminal state or invalid context" << std::endl;
        return;
    }

    if (state.tokens_sequence.empty()) {
        std::cout << "[MCTS] populate_untried_actions: Empty token sequence" << std::endl;
        return;
    }

    // Use the sampler to get properly processed candidates
    // Create a temporary sampler context for this specific position
    struct common_sampler * temp_sampler = common_sampler_clone(l_sampler);
    
    // Accept all tokens in the current sequence to set up the sampler state
    for (int token : state.tokens_sequence) {
        common_sampler_accept(temp_sampler, static_cast<llama_token>(token), false);
    }
    
    // Sample a token to trigger the sampling pipeline and get candidates
    // This will populate the internal candidates array with properly processed probabilities
    llama_token sample_token = common_sampler_sample(temp_sampler, l_ctx, -1);
    const auto * processed_candidates = common_sampler_get_candidates(temp_sampler);
    
    if (!processed_candidates || processed_candidates->size == 0) {
        common_sampler_free(temp_sampler);
        std::cout << "[MCTS] populate_untried_actions: No processed candidates" << std::endl;
        return;
    }

    // Select top candidates for MCTS exploration, avoiding problematic tokens
    const int top_n_for_mcts_actions = 10;
    int added_actions = 0;
    
    for (size_t i = 0; i < processed_candidates->size && added_actions < top_n_for_mcts_actions; ++i) {
        llama_token token_id = processed_candidates->data[i].id;
        
        // Skip tokens that would cause immediate termination unless they're reasonable
        // Skip EOS unless we have a reasonable sequence length
        if (token_id == llama_vocab_eos(l_vocab) && state.tokens_sequence.size() < 5) {
            continue;
        }
        
        // Skip very low probability tokens (less than 1% of max probability)
        if (i > 0 && processed_candidates->data[i].p < processed_candidates->data[0].p * 0.01f) {
            break;
        }
        
        // Skip special tokens that might cause issues (BOS, padding, etc.)
        if (token_id == llama_vocab_bos(l_vocab) || token_id == llama_vocab_pad(l_vocab)) {
            continue;
        }
        
        untried_actions.push_back(token_id);
        added_actions++;
    }
    
    common_sampler_free(temp_sampler);
    
    std::cout << "[MCTS] populate_untried_actions: Added " << untried_actions.size() << " potential actions" << std::endl;
    
    std::random_device rd_shuffle;
    std::mt19937 g_shuffle(rd_shuffle());
    std::shuffle(untried_actions.begin(), untried_actions.end(), g_shuffle);
}


bool MCTSNode::is_fully_expanded() const {
    return untried_actions.empty();
}

bool MCTSNode::is_terminal_node() const {
    return state.is_terminal_state(l_vocab, llama_n_ctx(l_ctx));
}

std::shared_ptr<MCTSNode> MCTSNode::uct_select_child(double exploration_constant) const {
    if (children.empty()) return nullptr;

    std::shared_ptr<MCTSNode> best_child = nullptr;
    double best_uct_value = -std::numeric_limits<double>::infinity();
    int unvisited_children_count = 0;
    for(const auto& child : children) if(child->n_visits == 0) unvisited_children_count++;

    for (const auto& child : children) {
        if (child->n_visits == 0) { // Prefer unvisited children
             // If multiple unvisited, UCT is effectively infinite.
             // A common strategy is to pick randomly among unvisited, or just the first.
             // Here, we'll pick the first unvisited one encountered.
            return child;
        }
        double uct_value = (child->q_value / child->n_visits) +
                           exploration_constant * std::sqrt(std::log(static_cast<double>(this->n_visits)) / child->n_visits);
        if (uct_value > best_uct_value) {
            best_uct_value = uct_value;
            best_child = child;
        }
    }
    return best_child;
}

std::shared_ptr<MCTSNode> MCTSNode::expand() {
    if (untried_actions.empty() || is_terminal_node()) {
        return nullptr;
    }

    int action_to_try = untried_actions.back();
    untried_actions.pop_back();

    GameState new_game_state = state;
    new_game_state.tokens_sequence.push_back(action_to_try);
    // Check if this new state is terminal (e.g. added EOS or reached max_length)
    new_game_state.is_terminal = new_game_state.is_terminal_state(l_vocab, llama_n_ctx(l_ctx));
    if (new_game_state.is_terminal && action_to_try == llama_vocab_eos(l_vocab)) {
        new_game_state.reward = 1.0f; // Example: positive reward for clean EOS
    } else if (new_game_state.is_terminal) {
        new_game_state.reward = 0.0f; // Example: neutral for max length
    }


    auto new_node = std::make_shared<MCTSNode>(new_game_state, l_ctx, l_model, l_vocab, l_sampler, shared_from_this());
    children.push_back(new_node);
    return new_node;
}

void MCTSNode::backpropagate(double reward_value) {
    n_visits++;
    q_value += reward_value;
    if (parent) {
        parent->backpropagate(reward_value);
    }
}

// --- MCTS Implementation ---
MCTS::MCTS(struct llama_context* ctx, struct llama_model* model, const struct llama_vocab* vocab, struct common_sampler* sampler, double ec)
    : l_ctx_(ctx), l_model_(model), l_vocab_(vocab), l_sampler_(sampler), exploration_constant_(ec) {
    std::random_device rd;
    random_engine_ = std::mt19937(rd());
    std::cout << "[MCTS] MCTS instance created with exploration constant: " << ec << std::endl;
}

std::shared_ptr<MCTSNode> MCTS::select_promising_node(std::shared_ptr<MCTSNode> root_node) {
    std::shared_ptr<MCTSNode> node = root_node;
    while (node && !node->is_terminal_node() && node->is_fully_expanded()) {
        node = node->uct_select_child(exploration_constant_);
    }
    return node ? node : root_node;
}

GameState MCTS::simulate_playout(std::shared_ptr<MCTSNode> node) {
    GameState current_playout_state = node->state;
    
    // Save l_ctx_ state before simulation
    size_t ctx_state_size = llama_state_get_size(l_ctx_);
    std::vector<uint8_t> ctx_state_mem(ctx_state_size);
    llama_state_get_data(l_ctx_, ctx_state_mem.data(), ctx_state_mem.size());

    // Create a fresh sampler for this simulation, inheriting parameters
    struct common_sampler * sim_sampler = common_sampler_clone(l_sampler_);
    for(int token : current_playout_state.tokens_sequence) { // Initialize sim_sampler with current sequence
        common_sampler_accept(sim_sampler, static_cast<llama_token>(token), false);
    }

    // Use a smaller batch size to respect limits - single token at a time for simulation
    llama_batch batch_sim = llama_batch_init(1, 0, 1);
    int sim_n_past = current_playout_state.tokens_sequence.size(); // n_past for the simulation context

    // Simulate from the current_playout_state
    // The KV cache in l_ctx_ should be at the state of current_playout_state.tokens_sequence
    // This is ensured by the expand->populate_untried_actions path or the root node setup.

    int simulation_depth = 0;
    const int max_simulation_depth = 20; // Limit simulation length

    while (!current_playout_state.is_terminal_state(l_vocab_, llama_n_ctx(l_ctx_)) && simulation_depth < max_simulation_depth) {
        common_batch_clear(batch_sim);
        if (current_playout_state.tokens_sequence.empty()) {
             current_playout_state.is_terminal = true; current_playout_state.reward = -1.0f; break;
        }
        
        // Decode only the last token of the current simulation sequence to get logits for the next.
        // The KV cache is advanced by this decode.
        common_batch_add(batch_sim, current_playout_state.tokens_sequence.back(), sim_n_past - 1, {0}, true);

        if (llama_decode(l_ctx_, batch_sim) != 0) {
            current_playout_state.is_terminal = true; current_playout_state.reward = -2.0f; break;
        }

        // Sample the next token using the proper sampling pipeline
        llama_token next_token = common_sampler_sample(sim_sampler, l_ctx_, 0);
        common_sampler_accept(sim_sampler, next_token, true);

        current_playout_state.tokens_sequence.push_back(next_token);
        sim_n_past++;
        simulation_depth++;

        if (next_token == llama_vocab_eos(l_vocab_)) {
            current_playout_state.is_terminal = true;
            current_playout_state.reward = 1.0f; // Reward for clean EOS
        } else if (current_playout_state.tokens_sequence.size() >= (size_t)llama_n_ctx(l_ctx_)) {
            current_playout_state.is_terminal = true;
            current_playout_state.reward = 0.0f; // Neutral for max length
        }
    }
    if (simulation_depth >= max_simulation_depth && !current_playout_state.is_terminal) {
        current_playout_state.is_terminal = true;
        current_playout_state.reward = -0.5f; // Penalize hitting max simulation depth without EOS
    }


    llama_batch_free(batch_sim);
    common_sampler_free(sim_sampler);
    
    // Restore l_ctx_ state after simulation
    llama_state_set_data(l_ctx_, ctx_state_mem.data(), ctx_state_mem.size());

    return current_playout_state;
}

void MCTS::backpropagate_rewards(std::shared_ptr<MCTSNode> node, double reward_value) {
    node->backpropagate(reward_value);
}

void MCTS::run_iteration(std::shared_ptr<MCTSNode> root_node) {
    std::shared_ptr<MCTSNode> promising_node = select_promising_node(root_node);
    if (!promising_node) return;

    std::shared_ptr<MCTSNode> expanded_node = promising_node;
    if (!promising_node->is_terminal_node() && !promising_node->is_fully_expanded()) {
        expanded_node = promising_node->expand();
        if (!expanded_node) {
            expanded_node = promising_node;
        }
    }
    
    GameState playout_result_state = simulate_playout(expanded_node);
    backpropagate_rewards(expanded_node, playout_result_state.reward);
}

int MCTS::get_best_action(std::shared_ptr<MCTSNode> root_node, int num_iterations) {
    std::cout << "[MCTS] get_best_action called with " << num_iterations << " iterations" << std::endl;
    
    if (!root_node || !l_ctx_ || !l_sampler_) {
        std::cout << "[MCTS] ERROR: Invalid MCTS setup - root_node: " << (root_node ? "valid" : "null")
                  << ", l_ctx_: " << (l_ctx_ ? "valid" : "null")
                  << ", l_sampler_: " << (l_sampler_ ? "valid" : "null") << std::endl;
        return -1;
    }

    // Save llama_context (KV cache) state
    size_t ctx_state_size = llama_state_get_size(l_ctx_);
    std::vector<uint8_t> ctx_state_mem(ctx_state_size);
    if (ctx_state_size > 0) { // Only copy if state size is non-zero
        llama_state_get_data(l_ctx_, ctx_state_mem.data(), ctx_state_mem.size());
    }


    // Save sampler state
    // Note: common_sampler doesn't have state save/restore functionality
    // We'll work with a cloned sampler instead


    std::cout << "[MCTS] Starting " << num_iterations << " MCTS iterations" << std::endl;
    for (int i = 0; i < num_iterations; ++i) {
        run_iteration(root_node);
        if (i % 10 == 0 || i < 5) {
            std::cout << "[MCTS] Completed iteration " << (i + 1) << "/" << num_iterations << std::endl;
        }
    }
    std::cout << "[MCTS] Completed all " << num_iterations << " iterations" << std::endl;

    // Restore sampler state
    // Remove this line since sampler_state_size is not defined
    {
        // Sampler state restoration not available, using cloned sampler approach
    }


    // Restore llama_context (KV cache) state
    if (ctx_state_size > 0) {
        llama_state_set_data(l_ctx_, ctx_state_mem.data(), ctx_state_mem.size());
    }


    if (root_node->children.empty()) {
        std::cout << "[MCTS] ERROR: Root node has no children after " << num_iterations << " iterations" << std::endl;
        return -1;
    }
    
    std::shared_ptr<MCTSNode> best_child = nullptr;
    int max_visits = -1;

    for (const auto& child : root_node->children) {
        if (child->n_visits > max_visits) {
            max_visits = child->n_visits;
            best_child = child;
        }
    }
    
    if (!best_child) {
        std::cout << "[MCTS] ERROR: Could not determine best child" << std::endl;
        return -1;
    }

    std::cout << "[MCTS] Best child found with " << best_child->n_visits << " visits, q_value: " << best_child->q_value << std::endl;
    std::cout << "[MCTS] Root state size: " << root_node->state.tokens_sequence.size()
              << ", Best child state size: " << best_child->state.tokens_sequence.size() << std::endl;

    if (best_child->state.tokens_sequence.size() > root_node->state.tokens_sequence.size()) {
        int selected_token = best_child->state.tokens_sequence[root_node->state.tokens_sequence.size()];
        std::cout << "[MCTS] Selected token: " << selected_token << std::endl;
        return selected_token;
    }

    std::cout << "[MCTS] ERROR: Best child's state not longer than root's" << std::endl;
    return -1;
}

} // namespace mcts