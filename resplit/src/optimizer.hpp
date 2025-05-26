#ifndef OPTIMIZER_H
#define OPTIMIZER_H

// Invocation Priorities
#define INFORMATION_PRIORITY 0
#define CONSTRAINT_PRIORITY 0
#define EXPLORATION_PRIORITY 0

#include <iostream>
#include <fstream>
#include <sstream>

#include <queue>

#include <chrono>
#include <unordered_map>
#include <tbb/tick_count.h>
#include <json/json.hpp>

// #include <alloca.h>

#include "configuration.hpp"
#include "dataset.hpp"
// #include "sorted_map.hpp"
#include "model.hpp"
#include "model_set.hpp"
#include "task.hpp"
#include "types.hpp"
#include "graph.hpp"
#include "queue.hpp"
#include "trie.hpp"
#include "memusage.h"
#include "integrity_violation.hpp"

using json = nlohmann::json;


class Optimizer {
public:
    Optimizer(void);
    ~Optimizer(void);

    void load(std::istream & data_source);

    void initialize(void);
    void reset(void);
    void reset_except_dataset(void);
    void set_rashomon_flag(void);
    void set_rashomon_bound(float);

    // @modifies lowerbound: the lowerbound on the global objective
    // @modifies upperbound: the upperbound on the global objective
    void objective_boundary(float * lowerbound, float * upperbound) const;

    // @returns the current difference between the global upperbound and the global lowerbound
    float uncertainty(void) const;

    // @returns true of the algorithm has reached a termination condition
    bool complete(void) const;

    // @returns the size fo the dependency graph
    unsigned int size(void) const;

    // @returns the real time spend in the optimization
    float elapsed(void) const;

    // @returns true if the configured time limit has been reached
    bool timeout(void) const;

    // @param id: ID of the requesting worker thread
    // @returns true if an update occured to the global objective boundary
    bool iterate(unsigned int id);

    // @modifies results: stores all potentially optimal models in results
    // @note: if the global optimality gap is non-zero, then results contains only models that fall within the optimality gap
    // @note: if the global optimality gap is non-zero, there is no gaurantee that results necessarily contains the optimal model
    void models(std::unordered_set< Model > & results);

    // @modifies results: stores all models in Rashomon set in results
    void rash_models(results_t & results);

    // Generates snapshot data for trace visualization
    void diagnostic_trace(int iteration, key_type const & focal_point);
    // Generates snapshot data for trace-tree visualization
    void diagnostic_tree(int iteration);

    // Print diagnostic trace for detected non-convergence of algorithm
    // Non-convergence is defined as the algorithm not terminating when it should have
    void diagnose_non_convergence(void); 

    // Print diagnositic trace for detected false-convergence of algorithm
    // False-convergence is defined as a premature termination of the algorithm
    void diagnose_false_convergence(void);
private:

    // Timing State
    tbb::tick_count start_time; // starting time of optimization
    unsigned long ticks = 0; // Number of ticks passed
    unsigned long tick_duration = 10000; // Number of iterations per tick
    bool active = true; // Flag indicating whether the optimization is still active
    bool rashomon_flag = false; // Flag for Rashomon

    // Analytics State
    Tile root; // Root indicator
    std::vector<int> translator; // Root indicator

    float global_boundary = std::numeric_limits<float>::max(); // Global optimality gap
    float global_upperbound = std::numeric_limits<float>::max(); // Global upperbound of the objective
    float global_lowerbound = -std::numeric_limits<float>::max(); // Global lowerbound of the objective
    std::vector< unsigned int > work_distribution; // Distribution of work done for each percentile
    unsigned int explore = 0.0; // Distributtion of work from downward message
    unsigned int exploit = 0.0; // Distribution of work from upward message
    float rashomon_bound = std::numeric_limits<float>::max(); // Bound to join the Rashomon set or not
    bool model_limit_exceeded = false; // Indicate whether the maximum model limit is exceeded
    long long unsigned int models_calls = 0; // Number of calls to the models() function 
    long long unsigned int re_explore_by_scope_update_count = 0; // Number of calls to the models() function 
    long long unsigned int re_explore_count = 0; // Number of calls to the models() function 
    long long unsigned int pruned_combinations_with_scope = 0; // Pruned combination through comparing combined obj val with scope
    long long unsigned int pruned_leaves_with_scope = 0; // Pruned leaves using the scope variable
    long long unsigned int pruned_trivial_extension = 0; // Pruned leaves using the scope variable
    std::size_t max_result_size = 0; // Maximum result size to ensure no overflow is happening

    // For exporting trie based on memory usage 
    int exported_idx = 0;

    // @param message: message to handle
    // @param id: id of the worker thread that is handling this message
    // @returns true if the optimization is still active
    bool dispatch(Message const & message, unsigned int id);

    bool store_self(Tile const & identifier, Task const & task, vertex_accessor & self);

    void store_children(Task & task, unsigned int id);

    void link_to_parent(Tile const & parent, Bitmask const & features, Bitmask const & signs, float scope, Tile const & self, translation_type const & order, adjacency_accessor & parents);

    void signal_exploiters(adjacency_accessor & parents, Task & self, unsigned int id);

    bool load_children(Task & task, Bitmask const & features, unsigned int id);

    bool load_parents(Tile const & identifier, adjacency_accessor & parents);

    bool load_self(Tile const & identifier, vertex_accessor & self);

    bool update_root(float lower, float upper);

    // @param set: identifier for the root node from which to extract optimal models
    // @modifies results: internal set of extracted models
    void models(key_type const & identifier, std::unordered_set< std::shared_ptr<Model> > & results, float scope = 0);

    void models_inner(key_type const & identifier, std::unordered_set< std::shared_ptr<Model> > & results, float scope = 0);

    void rash_models(key_type const & identifier, results_t & results, float scope = 0);

    void rash_models_inner(key_type const & identifier, results_t & results, float scope = 0);

    void insert_leaf_to_results(results_t & results, model_set_p & model);

    void print(void) const;
    void profile(void);
    
    // Export models based on current adjacency graph
    void export_models(std::string suffix);

    // Diagnostics
    bool diagnose_non_convergence(key_type const & set);
    bool diagnose_false_convergence(key_type const & set);
    bool diagnostic_trace(key_type const & identifier, json & tracer, key_type const & focal_point);
    bool diagnostic_tree(key_type const & identifier, json & tracer);
};

#endif