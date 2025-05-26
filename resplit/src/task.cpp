#include "task.hpp"
#include "graph.hpp"
#include "state.hpp"

Task::Task(void) {}

Task::Task(Bitmask const & capture_set, Bitmask const & feature_set, unsigned int id, bool rashomon_flag) {
    this -> _capture_set = capture_set;
    this -> _feature_set = feature_set;
    this -> _support = (float)(capture_set.count()) / (float)(State::dataset.height());
    float const regularization = Configuration::regularization;
    bool terminal = (this -> _capture_set.count() <= 1) || (this -> _feature_set.empty());

    float potential, min_loss, max_loss;
    unsigned int target_index;
    // Careful, the following method modifies capture_set
    State::dataset.summary(this -> _capture_set, this -> _information, potential, min_loss, max_loss, target_index, id);

    this -> _base_objective = max_loss + regularization;
    // Add lambda because we know this has at least 2 leaves
    float const lowerbound = std::min(this -> _base_objective, min_loss + 2 * regularization);
    float const upperbound = this -> _base_objective;

    // false && rashomon_flag
    if (rashomon_flag) {
        this -> _lowerbound = lowerbound;
        this -> _upperbound = upperbound;
    } else {
        if ( (1.0 - min_loss < regularization ) // Insufficient maximum accuracy
            || ( potential < 2 * regularization && (1.0 - max_loss) < regularization) ) // Leaf Support + Incremental Accuracy
        { // Insufficient support and leaf accuracy
            // Node is provably not part of any optimal tree
            this -> _lowerbound = this -> _base_objective;
            this -> _upperbound = this -> _base_objective;
            this -> _feature_set.clear();
        } else if (Configuration::cart_lookahead_depth != 0 && capture_set.get_depth_budget() == 1) {
            // Set cart depth to max(1, depth_budget - cart_lookahead_depth), while being careful with unsigned chars
            unsigned char depth_for_cart = 1;
            if (Configuration::cart_lookahead_depth < Configuration::depth_budget) {
                depth_for_cart = Configuration::depth_budget - Configuration::cart_lookahead_depth + 1;
            }
            this -> _lowerbound = cart(this -> _capture_set, this -> _feature_set, id, depth_for_cart);
            this -> _upperbound = this -> _lowerbound;
            this -> _feature_set.clear();
        }else if (
            max_loss - min_loss < regularization // Accuracy
            || potential < 2 * regularization // Leaf Support
            || terminal
        ) {
            // Node is provably not an internal node of any optimal tree
            this -> _lowerbound = this -> _base_objective;
            this -> _upperbound = this -> _base_objective;
            this -> _feature_set.clear();
            
        } else {
            // Node can be either an internal node or leaf
            this -> _lowerbound = lowerbound;
            this -> _upperbound = upperbound;
        }
    }

    if (Configuration::depth_budget != 0 && capture_set.get_depth_budget() == 1) {
        // we are using depth constraints, and depth budget is exhausted
        this -> _lowerbound = this -> _base_objective;
        this -> _upperbound = this -> _base_objective;
        this -> _feature_set.clear();
    }

    if (this -> _lowerbound > this -> _upperbound) {
        std::stringstream reason;
        reason << "Invalid Lowerbound (" << this -> _lowerbound << ") or Upperbound (" << this -> _upperbound << ")." << std::endl;
        throw IntegrityViolation("Task::Task", reason.str());
    }
}

float Task::support(void) const { return this -> _support; }

float Task::information(void) const { return this -> _information; }

float Task::base_objective(void) const { return this -> _base_objective; }

float Task::uncertainty(void) const { return std::max((float)(0.0), upperbound() - lowerbound()); }

float Task::lowerbound(void) const { return this -> _lowerbound; }
float Task::upperbound(void) const { return this -> _upperbound; }
float Task::lowerscope(void) const { return this -> _lowerscope; }
float Task::upperscope(void) const { return this -> _upperscope; }
float Task::rashomon_bound(void) const { return this -> _rashomon_bound; }
void Task::set_rashomon_flag(void) { _rashomon_flag = true; return;}
void Task::set_rashomon_bound(float bound) { //std::cout << "set_bound: " << bound << std::endl; 
_rashomon_bound = bound; }


Bitmask const & Task::capture_set(void) const { return this -> _capture_set; }
Bitmask const & Task::feature_set(void) const { return this -> _feature_set; }
Tile & Task::identifier(void) { return this -> _identifier; }
std::vector<int> & Task::order(void) { return this -> _order; }

void Task::scope(float new_scope) {
    if (new_scope == 0) { return; }
    new_scope = std::max((float)(0.0), new_scope);
    this -> _upperscope = this -> _upperscope == std::numeric_limits<float>::max() ? new_scope : std::max(this -> _upperscope, new_scope);
    this -> _lowerscope = this -> _lowerscope == -std::numeric_limits<float>::max() ? new_scope : std::min(this -> _lowerscope, new_scope);
}

void Task::prune_feature(unsigned int index) { this -> _feature_set.set(index, false); }

void Task::create_children(unsigned int id) {    
    // this -> _lowerbound = this -> _base_objective;
    // this -> _upperbound = this -> _base_objective;
    Bitmask & buffer = State::locals[id].columns[0];
    bool conditions[2] = {false, true};
    Bitmask const & features = this -> _feature_set;
    for (int j_begin = 0, j_end = 0; features.scan_range(true, j_begin, j_end); j_begin = j_end) {
        for (int j = j_begin; j < j_end; ++j) {
            bool skip = false;
            for (unsigned int k = 0; k < 2; ++k) {
                buffer = this -> _capture_set;
                State::dataset.subset(j, conditions[k], buffer);
                unsigned int buffer_count = buffer.count();
                if (std::min(buffer_count, buffer.size() - buffer_count) <= Configuration::minimum_captured_points) {
                    skip = true;
                    continue;
                }
                Task child(buffer, this -> _feature_set, id);
                State::locals[id].neighbourhood[2 * j + k] = child;
            }
            if (skip) { prune_feature(j); }


            // Task & left = State::locals[id].neighbourhood[2 * j];
            // Task & right = State::locals[id].neighbourhood[2 * j + 1];

        }
    }
}

void Task::prune_features(unsigned int id) {
    if (Configuration::continuous_feature_exchange) { continuous_feature_exchange(id); }
    if (Configuration::feature_exchange) { feature_exchange(id); }

    // this -> _lowerbound = this -> _base_objective;
    // this -> _upperbound = this -> _base_objective;
    Bitmask & buffer = State::locals[id].columns[0];
    bool conditions[2] = {false, true};
    Bitmask const & features = this -> _feature_set;
    int optimal_feature = -1;
    float new_lower = this -> _base_objective;
    float new_upper = this -> _base_objective;
    for (int j_begin = 0, j_end = 0; features.scan_range(true, j_begin, j_end); j_begin = j_end) {
        for (int j = j_begin; j < j_end; ++j) {
            float lower = 0.0, upper = 0.0;
            
            Task & left = State::locals[id].neighbourhood[2 * j];
            Task & right = State::locals[id].neighbourhood[2 * j + 1];

            if (Configuration::rule_list) {
                float lower_negative = left.base_objective() + right.lowerbound();
                float lower_positive = left.lowerbound() + right.base_objective();
                lower = std::min(lower_negative, lower_positive);
                float upper_negative = left.base_objective() + right.upperbound();
                float upper_positive = left.upperbound() + right.base_objective();
                upper = std::min(upper_negative, upper_positive);
            } else {
                lower = left.lowerbound() + right.lowerbound();
                upper = left.upperbound() + right.upperbound();
            }

            // std::cout << "Split: " << j << ", Bound: [" << left.lowerbound() << " + " << right.lowerbound()  << ", " << left.upperbound() << " + " << right.upperbound() << "]" << std::endl;
            // std::cout << "Split: " << j << ", Bound: [" << lower  << ", " << upper << "]" << std::endl;

            // if (this -> _rashomon_flag) { 
            //     if (lower > this -> _rashomon_bound) { continue; } // Hierarchical objective lower bounds for Rashomon set
            // } else if (lower > this -> _upperscope) { continue; } // Hierarchical objective lower bounds
            if (lower > this -> _upperscope) { continue; } // Hierarchical objective lower bounds
            if (upper < new_upper) { optimal_feature = j; }
            new_lower = std::min(new_lower, lower);
            new_upper = std::min(new_upper, upper);
        }
    }
    if (new_lower > this -> _upperscope) { return; }
    this -> _lowerbound = new_lower;
    this -> _upperbound = new_upper;
    this -> _optimal_feature = optimal_feature;
}

void Task::continuous_feature_exchange(unsigned int id) {
    Bitmask const & features = this -> _feature_set;
    for (auto it = State::dataset.encoder.boundaries.begin(); it != State::dataset.encoder.boundaries.end(); ++it) {
        int start = it -> first, finish = it -> second;
        for (int i = features.scan(start, true), j = features.scan(i + 1, true); j < finish; i = j, j = features.scan(j + 1, true)) {
            float alpha = State::locals[id].neighbourhood[2 * i].lowerbound();
            float beta = State::locals[id].neighbourhood[2 * j].upperbound();
            if (alpha >= beta) { prune_feature(i); }
            if (j >= finish - 1) { break; }
        }

        for (int i = features.rscan(finish - 1, true), j = features.rscan(i - 1, true); j >= start; i = j, j = features.rscan(j - 1, true)) {
            float alpha = State::locals[id].neighbourhood[2 * i + 1].lowerbound();
            float beta = State::locals[id].neighbourhood[2 * j + 1].upperbound();
            if (alpha >= beta) { prune_feature(i); }
            if (j <= start) { break; }
        }
    }
}

void Task::feature_exchange(unsigned int id) {
    Bitmask const & features = this -> _feature_set;
    int max = features.size();
    Bitmask & buffer = State::locals[id].columns[0];
    for (int i = features.scan(0, true); i < max; i = features.scan(i + 1, true)) {
        for (int j = features.scan(0, true); j < max; j = features.scan(j + 1, true)) {
            if (i == j) { continue; }
            for (unsigned short k = 0; k < 4; ++k) {
                buffer = this -> _capture_set;
                bool i_sign = (bool)(k & 1);
                bool j_sign = (bool)((k >> 1) & 1);
                State::dataset.subset(i, i_sign, buffer); // population after applying i filter
                int i_count = buffer.count(); 
                State::dataset.subset(j, j_sign, buffer); // population remaining if !j filter is applied
                if (i_count != buffer.count()) { continue; } // implies that i is not a subset of j
                // implies that i IS a subset of j, therefore !j is a subset of !i
                // (since i + !i covers the same set as j + !j)
                float not_i_risk = State::locals[id].neighbourhood[2 * i + (int)(!i_sign)].upperbound();
                float not_j_risk = State::locals[id].neighbourhood[2 * j + (int)(!j_sign)].lowerbound();
                // not_i_risk <= not_j_risk and i IS a subset of j implies that risk_i <= risk_j
                if (not_i_risk <= not_j_risk && features.get(i)) { prune_feature(j); break; }
            }
        }
    }
}

void Task::send_explorers(float new_scope, unsigned int id) {
    if (!_rashomon_flag && this -> uncertainty() == 0) { return; }
    this -> scope(new_scope);

    float exploration_boundary;
    if (this->_rashomon_flag){
        exploration_boundary = rashomon_bound();
    }
    else {
        exploration_boundary = upperbound();
    }

    if (Configuration::look_ahead) { exploration_boundary = std::min(exploration_boundary, this -> _upperscope); }

    Bitmask const & features = this -> _feature_set;
    for (int j_begin = 0, j_end = 0; features.scan_range(true, j_begin, j_end); j_begin = j_end) {
        for (unsigned int j = j_begin; j < j_end; ++j) {
            Task & left = State::locals[id].neighbourhood[2 * j];
            Task & right = State::locals[id].neighbourhood[2 * j + 1];
            float lower, upper;
            if (Configuration::rule_list) {
                lower = std::min(left.lowerbound() + right.base_objective(), left.base_objective() + right.lowerbound());
                upper = std::min(left.upperbound() + right.base_objective(), left.base_objective() + right.upperbound());
            } else {
                lower = left.lowerbound() + right.lowerbound();
                upper = left.upperbound() + right.upperbound();
            }

            // additional requirement for skipping covered tasks. covered tasks must be unscoped:
            // that is, their upperbound must be strictly less than their scope 

            if (lower > exploration_boundary) { continue; } // Skip children that are out of scope 


            if (!_rashomon_flag && upper <= this -> _coverage) { continue; } // Skip children that have been explored

            if (Configuration::rule_list) {
                send_explorer(left, exploration_boundary - right.base_objective(), -(j + 1), id);
                send_explorer(right, exploration_boundary - left.base_objective(), (j + 1), id);
            } else {
                send_explorer(left, exploration_boundary - right.lowerbound(), -(j + 1), id);
                send_explorer(right, exploration_boundary - left.lowerbound(), (j + 1), id);
            }
        }
    }
    this -> _coverage = this -> _upperscope;
}

void Task::send_explorer(Task const & child, float scope, int feature, unsigned int id) {
    bool send = true;
    child_accessor key;

    if (State::graph.children.find(key, std::make_pair(this -> identifier(), feature))) {
        vertex_accessor child;
        State::graph.vertices.find(child, key -> second);

        if (scope < child -> second.upperscope()) {
            adjacency_accessor parents;
            State::graph.edges.find(parents, child -> second.identifier()); // insert backward look-up entry
            std::pair<adjacency_iterator, bool> insertion = parents -> second.insert(
                std::make_pair(this -> identifier(), std::make_pair(Bitmask(State::dataset.width(), false), scope)));
            insertion.first -> second.first.set(std::abs(feature) - 1, true);
            insertion.first -> second.second = std::min(insertion.first -> second.second, scope);
            child -> second.scope(scope);
            send = false;
        }
        key.release();
    }
    if (send) {
        State::locals[id].outbound_message.exploration(
            this->_identifier,  // sender tile
            child._capture_set, // recipient capture_set
            this->_feature_set, // recipient feature_set
            feature,            // feature
            scope,              // scope
            this->_support - this->_lowerbound); // priority
        State::queue.push(State::locals[id].outbound_message);
    }
}


bool Task::update(float lower, float upper, int optimal_feature) {
    bool change = lower != this -> _lowerbound || upper != this -> _upperbound;
    this -> _lowerbound = std::max(this -> _lowerbound, lower);
    // if (this->_rashomon_flag) { this -> _upperbound = this->_rashomon_bound;}
    // else { this -> _upperbound = std::min(this -> _upperbound, upper); }
    this -> _upperbound = std::min(this -> _upperbound, upper); 
    this -> _lowerbound = std::min(this -> _upperbound, this -> _lowerbound);

    this -> _optimal_feature = optimal_feature;

    float regularization = Configuration::regularization;
    if ((Configuration::cancellation && 1.0 - this -> _lowerbound < 0.0)
        || this -> _upperbound - this -> _lowerbound <= std::numeric_limits<float>::epsilon()) {
        this -> _lowerbound = this -> _upperbound;
    }
    return change;
}

std::string Task::inspect(void) const {
    std::stringstream status;
    status << "Capture: " << this -> _capture_set.to_string() << std::endl;
    // status << "  State[SEDRCIT] = " << (int)(sampled()) << (int)(explored()) << (int)(delegated()) << (int)(resolved()) << (int)(cancelled()) << (int)(informed()) << (int)(terminal()) << std::endl;
    status << "  Base: " << this -> _base_objective << ", Bound: [" << this -> _lowerbound << ", " << this -> _upperbound << "]" << std::endl;
    status << "  Coverage: " << this -> _coverage << ", Scope: [" << this -> _lowerscope << ", " << this  -> _upperscope << "]" << std::endl;
    status << "  Feature: " << this -> _feature_set.to_string() << std::endl;
    return status.str();
}

// expects depth to be at least 1. If depth is 1, the function will make no further splits
float Task::cart(Bitmask const & capture_set, Bitmask const & feature_set, unsigned int id, unsigned char depth) const {
    Bitmask left(State::dataset.height());
    Bitmask right(State::dataset.height());
    float potential, min_loss, max_loss, base_info;
    unsigned int target_index;
    State::dataset.summary(capture_set, base_info, potential, min_loss, max_loss, target_index, id);
    float base_risk = max_loss + Configuration::regularization;

    if (depth == 1
        || max_loss - min_loss < Configuration::regularization
        || 1.0 - min_loss < Configuration::regularization
        || (potential < 2 * Configuration::regularization && (1.0 - max_loss) < Configuration::regularization
        || feature_set.empty())) {
        return base_risk;
    }

    int information_maximizer = -1;
    float information_gain = 0;
    for (int j_begin = 0, j_end = 0; feature_set.scan_range(true, j_begin, j_end); j_begin = j_end) {
        for (int j = j_begin; j < j_end; ++j) {
            float left_info, right_info;
            left = capture_set;
            right = capture_set;
            State::dataset.subset(j, false, left);
            State::dataset.subset(j, true, right);

            if (left.empty() || right.empty()) { continue; }

            State::dataset.summary(capture_set, left_info, potential, min_loss, max_loss, target_index, id);
            State::dataset.summary(capture_set, right_info, potential, min_loss, max_loss, target_index, id);

            float gain = left_info + right_info - base_info;
            if (gain > information_gain) {
                information_maximizer = j;
                information_gain = gain;
            }
        }
    }

    if (information_maximizer == -1) { return base_risk; }

    left = capture_set;
    right = capture_set;
    State::dataset.subset(information_maximizer, false, left);
    State::dataset.subset(information_maximizer, true, right);
    float risk = cart(left, feature_set, id, depth - 1) + cart(right, feature_set, id, depth - 1);
    return std::min(risk, base_risk);
}