#include "optimizer.hpp"

#include "optimizer/diagnosis/false_convergence.hpp"
#include "optimizer/diagnosis/non_convergence.hpp"
#include "optimizer/diagnosis/trace.hpp"
#include "optimizer/diagnosis/tree.hpp"
#include "optimizer/dispatch/dispatch.hpp"
#include "optimizer/extraction/models.hpp"
#include "optimizer/extraction/rash_models.hpp"

Optimizer::Optimizer(void) {
    return;
}

Optimizer::~Optimizer(void) {
    State::reset();
    return;
}



void Optimizer::load(std::istream & data_source) { State::initialize(data_source, Configuration::worker_limit); }

void Optimizer::reset(void) { State::reset(); }

void Optimizer::reset_except_dataset(void) { 
    active = true;
    State::reset_except_dataset();
}

void Optimizer::set_rashomon_flag(void) { this -> rashomon_flag = true; }
void Optimizer::set_rashomon_bound(float bound) { this -> rashomon_bound = bound; }


void Optimizer::initialize(void) {
    // Initialize Profile Output
    if (Configuration::profile != "") {
        std::ofstream profile_output(Configuration::profile);
        profile_output << "iterations,time,lowerbound,upperbound,graph_size,queue_size,explore,exploit";
        profile_output << std::endl;
        profile_output.flush();
    }

    // Initialize Timing State
    this -> start_time = tbb::tick_count::now();

    int const n = State::dataset.height();
    int const m = State::dataset.width();
    // Enqueue for exploration: 
    //  - First, initialize depth for root
    unsigned char depth_budget = 0;
    if (Configuration::cart_lookahead_depth != 0){ 
        depth_budget = Configuration::cart_lookahead_depth;
    }else{ 
        depth_budget = Configuration::depth_budget;
    }
    //  - Then, enqueue that root task
    State::locals[0].outbound_message.exploration(Tile(), Bitmask(n, true, NULL, depth_budget), Bitmask(m, true), 0, std::numeric_limits<float>::max());
    State::queue.push(State::locals[0].outbound_message);

    return;
}


void Optimizer::objective_boundary(float * lowerbound, float * upperbound) const {
    * lowerbound = this -> global_lowerbound;
    * upperbound = this -> global_upperbound;

}

float Optimizer::uncertainty(void) const {
    float const epsilon = std::numeric_limits<float>::epsilon();
    float value = this -> global_upperbound - this -> global_lowerbound;
    return value < epsilon ? 0 : value;
}

float Optimizer::elapsed(void) const {
    auto now = tbb::tick_count::now();
    float duration = (now - this -> start_time).seconds();
    return duration;
}

bool Optimizer::timeout(void) const {
    return (Configuration::time_limit > 0 && elapsed() > Configuration::time_limit);
}

bool Optimizer::complete(void) const {
    return uncertainty() == 0;
}

unsigned int Optimizer::size(void) const {
    return State::graph.size();
}

bool Optimizer::iterate(unsigned int id) {
    bool update = false;

    if (State::queue.pop(State::locals[id].inbound_message)) {
        update = dispatch(State::locals[id].inbound_message, id);
        switch (State::locals[id].inbound_message.code) {
            case Message::exploration_message: { this -> explore += 1; break; }
            case Message::exploitation_message: { this -> exploit += 1; break; }
        }
    }
    // Worker 0 is responsible for managing ticks and snapshots
    if (id == 0) {
        this -> ticks += 1;

        // snapshots that would need to occur every iteration
        // if (Configuration::trace != "") { this -> diagnostic_trace(this -> ticks, State::locals[id].message); }
        if (Configuration::tree != "") { this -> diagnostic_tree(this -> ticks); }

        // snapshots that can skip unimportant iterations
        if (update || complete() || ((this -> ticks) % (this -> tick_duration)) == 0) { // Periodic check for completion for timeout
            // Update the continuation flag for all threads
            this -> active = !complete() && !timeout() && (Configuration::worker_limit > 1 || State::queue.size() > 0);
            this -> print();
            this -> profile();
        }
        
        std::vector<int> memory_checkpoint = Configuration::memory_checkpoints;
        if (rashomon_flag && exported_idx < memory_checkpoint.size() && getCurrentRSS() > memory_checkpoint[exported_idx] * 1000000) {
            export_models(std::to_string(memory_checkpoint[exported_idx]));
            exported_idx++;
            std::cout << "Memory usage after extraction: " << getCurrentRSS() / 1000000 << std::endl;
        }
    }
    return this -> active;
}

void Optimizer::print(void) const {
    if (Configuration::verbose) { // print progress to standard output
        float lowerbound, upperbound;
        objective_boundary(& lowerbound, & upperbound);
        std::cout <<
            "Time: " << elapsed() <<
            ", Objective: [" << lowerbound << ", " << upperbound << "]" <<
            ", Boundary: " << this -> global_boundary <<
            ", Graph Size: " << State::graph.size() <<
            ", Queue Size: " << State::queue.size() << std::endl;
    }
}

void Optimizer::profile(void) {
    if (Configuration::profile != "") {
        std::ofstream profile_output(Configuration::profile, std::ios_base::app);
        float lowerbound, upperbound;
        objective_boundary(& lowerbound, & upperbound);
        profile_output << this -> ticks << "," << elapsed() << "," <<
            lowerbound << "," << upperbound << "," << State::graph.size() << "," << 
            State::queue.size() << "," << this -> explore << "," << this -> exploit;
        profile_output << std::endl;
        profile_output.flush();
        this -> explore = 0;
        this -> exploit = 0;
    }
}

void Optimizer::export_models(std::string suffix) {
    if (Configuration::rashomon_trie != "") {
        std::unordered_set< Model > models;
        this->models(models);
        bool calculate_size = false;
        char const *type = "node";
        Trie* tree = new Trie(calculate_size, type);
        tree->insert_root();
        for (auto iterator = models.begin(); iterator != models.end(); ++iterator) {
            tree->insert_model(&(*iterator));
        }

        std::string serialization;
        tree->serialize(serialization, 2);
        // 
        std::stringstream fmt;
        fmt << Configuration::rashomon_trie << "-" << suffix;
        std::string file_name = fmt.str();

        if(Configuration::verbose) { std::cout << "Storing Models in: " << file_name << std::endl; }
        std::ofstream out(file_name);
        out << serialization;
        out.close();
        
        State::graph.models.clear();
    }
}
