#include "optimizer.hpp"

#include <utility>

#include "configuration.hpp"
#include "dataset.hpp"

Optimizer::Optimizer(const Configuration &config, const Dataset &dataset)
    : m_config(config), m_dataset(dataset), m_local_states(m_config.worker_limit) {
    for (auto &local_state : m_local_states) {
        local_state.initialize(m_dataset.m_number_rows, m_dataset.m_number_features, m_dataset.m_number_targets);
    }
}

void Optimizer::initialize() {
    // Initialize Profile Output
    if (!m_config.profile.empty()) {
        std::ofstream profile_output(m_config.profile);
        profile_output << "iterations,time,lower_bound,upper_bound,graph_size,queue_size,explore,exploit";
        profile_output << std::endl;
        profile_output.flush();
    }

    int const n = m_dataset.m_number_rows;
    int const m = m_dataset.m_number_features;

    // Enqueue for exploration
    // first, Initialize depth for root
    unsigned char depth_budget = 0;
    if (m_config.cart_lookahead_depth != 0){ 
        depth_budget = m_config.cart_lookahead_depth;
    }else{ 
        depth_budget = m_config.depth_budget;
    }
    // Then, enqueue that root task
    m_local_states[0].outbound_message.exploration(Bitmask(), Bitmask(n, true, nullptr, depth_budget),
                                                   Bitmask(m, true), 0, std::numeric_limits<float>::max());
    m_queue.push(m_local_states[0].outbound_message);

    // Initialize timing state
    m_start_time = high_resolution_clock::now();
}

std::pair<double, double> Optimizer::objective_boundary() const {
    return {this->global_lowerbound, this->global_upperbound};
}

void Optimizer::objective_boundary(float *lowerbound, float *upperbound) const {
    *lowerbound = this->global_lowerbound;
    *upperbound = this->global_upperbound;
}

float Optimizer::uncertainty(void) const {
    float const epsilon = std::numeric_limits<float>::epsilon();
    float value = this->global_upperbound - this->global_lowerbound;
    return value < epsilon ? 0 : value;
}

double Optimizer::time_elapsed() const {
    auto now = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(now - m_start_time).count();
    return static_cast<double>(duration) / 1000.0;
}

bool Optimizer::timeout(void) const { return (m_config.time_limit > 0 && time_elapsed() > m_config.time_limit); }

bool Optimizer::complete(void) const { return uncertainty() == 0; }

unsigned int Optimizer::size(void) const { return m_graph.size(); }

bool Optimizer::iterate(unsigned int id) {
    bool update = false;
    if (m_queue.pop(m_local_states[id].inbound_message)) {
        update = dispatch(m_local_states[id].inbound_message, id);
        switch (m_local_states[id].inbound_message.code) {
            case Message::exploration_message: {
                this->explore += 1;
                break;
            }
            case Message::exploitation_message: {
                this->exploit += 1;
                break;
            }
        }
    }

    // Worker 0 is responsible for managing ticks and snapshots
    if (id == 0) {
        this->ticks += 1;

        // snapshots that would need to occur every iteration
        // if (Configuration::trace != "") { this -> diagnostic_trace(this ->
        // ticks, state.locals[id].message); }
        if (!m_config.tree.empty()) {
            std::cout << "Diagnostic tree is no longer supported\n";
            exit(-1);
            // this -> diagnostic_tree(this -> ticks);
        }

        // snapshots that can skip unimportant iterations
        if (update || complete() ||
            ((this->ticks) % (this->tick_duration)) == 0) {  // Periodic check for completion for timeout
            // Update the continuation flag for all threads
            this->active = !complete() && !timeout() && (m_config.worker_limit > 1 || m_queue.size() > 0);
            this->print();
            this->profile();
        }
    }
    return this->active;
}

void Optimizer::print(void) const {
    if (m_config.verbose) {  // print progress to standard output
        float lowerbound, upperbound;
        objective_boundary(&lowerbound, &upperbound);
        std::cout << "Time: " << time_elapsed() << ", Objective: [" << lowerbound << ", " << upperbound << "]"
                  << ", Boundary: " << this->global_boundary << ", Graph Size: " << m_graph.size()
                  << ", Queue Size: " << m_queue.size() << std::endl;
    }
}

void Optimizer::profile(void) {
    if (m_config.profile != "") {
        std::ofstream profile_output(m_config.profile, std::ios_base::app);
        float lowerbound, upperbound;
        objective_boundary(&lowerbound, &upperbound);
        profile_output << this->ticks << "," << time_elapsed() << "," << lowerbound << "," << upperbound << ","
                       << m_graph.size() << "," << m_queue.size() << "," << this->explore << "," << this->exploit;
        profile_output << std::endl;
        profile_output.flush();
        this->explore = 0;
        this->exploit = 0;
    }
}

void Optimizer::send_explorers(Task &parent, float new_scope, unsigned int id) {
    if (parent.uncertainty() == 0) {
        return;
    }
    parent.scope(new_scope);

    float exploration_boundary = parent.upperbound();
    if (m_config.look_ahead) {
        exploration_boundary = std::min(exploration_boundary, parent.upperscope());
    }

    const Bitmask &features = parent.feature_set();
    for (int j_begin = 0, j_end = 0; features.scan_range(true, j_begin, j_end); j_begin = j_end) {
        for (unsigned int j = j_begin; j < j_end; ++j) {
            Task &left = m_local_states[id].neighbourhood[2 * j];
            Task &right = m_local_states[id].neighbourhood[2 * j + 1];
            float lower, upper;
            if (m_config.rule_list) {
                lower =
                    std::min(left.lowerbound() + right.base_objective(), left.base_objective() + right.lowerbound());
                upper =
                    std::min(left.upperbound() + right.base_objective(), left.base_objective() + right.upperbound());
            } else {
                lower = left.lowerbound() + right.lowerbound();
                upper = left.upperbound() + right.upperbound();
            }

            // additional requirement for skipping covered tasks. covered tasks
            // must be unscoped: that is, their upperbound must be strictly less
            // than their scope

            if (lower > exploration_boundary) {
                continue;
            }  // Skip children that are out of scope
            if (upper <= parent.coverage()) {
                continue;
            }  // Skip children that have been explored

            if (m_config.rule_list) {
                send_explorer(parent, left, exploration_boundary - right.base_objective(), -(j + 1), id);
                send_explorer(parent, right, exploration_boundary - left.base_objective(), (j + 1), id);
            } else {
                send_explorer(parent, left, exploration_boundary - right.guaranteed_lowerbound(m_config), -(j + 1), id);
                send_explorer(parent, right, exploration_boundary - left.guaranteed_lowerbound(m_config), (j + 1), id);
            }
        }
    }

    parent.set_coverage(parent.upperscope());
}

void Optimizer::send_explorer(Task &parent, const Task &child, float scope, int feature, unsigned int id) {
    bool send = true;
    child_accessor key;
    if (m_graph.children.find(key, std::make_pair(parent.capture_set(), feature))) {
        vertex_accessor child;
        m_graph.vertices.find(child, key->second);
        if (scope < child->second.upperscope()) {
            adjacency_accessor parents;
            m_graph.edges.find(parents,
                               child->second.capture_set());  // insert backward look-up entry
            std::pair<adjacency_iterator, bool> insertion = parents->second.insert(std::make_pair(
                parent.capture_set(), std::make_pair(Bitmask(m_dataset.m_number_features, false), scope)));
            insertion.first->second.first.set(std::abs(feature) - 1, true);
            insertion.first->second.second = std::min(insertion.first->second.second, scope);
            child->second.scope(scope);
            send = false;
        }
        key.release();
    }
    if (send) {
        m_local_states[id].outbound_message.exploration(parent.capture_set(),   // sender tile
                                                        child.capture_set(),   // recipient capture_set
                                                        parent.feature_set(),  // recipient feature_set
                                                        feature,               // feature
                                                        scope,                 // scope
                                                        parent.support() - parent.lowerbound());  // priority
        m_queue.push(m_local_states[id].outbound_message);
    }
}
