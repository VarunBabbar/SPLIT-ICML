#ifndef CONFIGURATION_H
#define CONFIGURATION_H

#include <fstream>
#include <string>

#include "nlohmann/json.hpp"

// Configuration object used to modify the algorithm behaviour
// By design, all running instances of the algorithm within the same process
// must share the same configuration
class Configuration {
    using json = nlohmann::json;

   public:
    Configuration() = default;

    friend std::ostream &operator<<(std::ostream &ostream, const Configuration &config);

    /// Constructs a new configuration object from a JSON object
    static Configuration from_json(const json& object);

    /// Dumps the configuration object to a JSON object
    json to_json() const;

    /// Save and Load the configuration object to a file
    void save(const std::string &path) const;
    static Configuration load(const std::string &path);

   public:
    // #TODO(Ilias): Replace the original configuration with these capitalized/renamed
    //               variables.
    //
    // /// Boolean flag to enable the upward propagation of cancelled sub-problems
    // bool CANCELLATION = true;
    //
    // /// The maximum tree depth for solutions. We denote a Tree with just a root
    // /// node as a Tree of depth 1. 0 is used as a sentinel value indicating a
    // /// lack of depth budget.
    // static const unsigned DEPTH_BUDGET_DISABLED = 0;
    // unsigned DEPTH_BUDGET = DEPTH_BUDGET_DISABLED;
    //
    // /// Boolean flag to enable the printing of diagnostic data used to detect
    // /// algorithm non-convergence.
    // bool DIAGNOSTICS = false;
    //
    // /// Boolean flag to enable the one step look ahead bound implemented via
    // /// scopes.
    // bool LOOK_AHEAD = true;
    //
    // /// The maximum number of optimal models to extract.
    // unsigned MODEL_LIMIT = 1;
    //
    // /// Boolean flag to enable non-binary encodings.
    // bool NON_BINARY = false;
    //
    // /// The regularization penaly incurred for each leaf in the model.
    // float REGULARIZATION = 0.05;
    //
    // /// Boolean flag to enable rule list constraints on models.
    // bool RULE_LIST = false;
    //
    // /// Boolean flag to enable the similar support bound implemennted via a
    // /// distance index.
    // bool SIMILAR_SUPPORT = true;
    //
    // /// The maximum allowed runtime in seconds. 0 is used as a sentinel value
    // /// indicating a lack of runtime limit.
    // static const unsigned TIME_LIMIT_DISABLED = 0;
    // static const unsigned TIME_LIMIT_HALF_HOUR = 3600;
    // unsigned TIME_LIMIT = TIME_LIMIT_HALF_HOUR;
    //
    // /// The maximum allowed gap in global optimality before the optimization can
    // /// terminate.
    // float UNCERTAINTY_TOLERANCE = 0;
    //
    // /// Reference upperbound on the root problem generated using a greedy model.
    // /// Used to prune insufficiently improved sub-problems.
    // float UPPERBOUND_GUESS = 0;
    //
    // /// Boolean flag to enable verbose printing to standard output. The purpose
    // /// of this is as a debugging tool.
    // bool VERBOSE = false;
    //
    // /// The maximum number of worker threads.
    // unsigned WORKER_LIMIT = 1;

    // TODO get rid of the rest of these.

    float regularization = 0.05;        // The penalty incurred for each leaf in the model
    float upperbound_guess = 0.0;       // Upperbound on the root problem for pruning
                                        // problems using a greedy model

    unsigned int time_limit = 0;       // The maximum allowed runtime (seconds). 0 means unlimited.
    unsigned int worker_limit = 1;     // The maximum allowed worker threads. 0
                                       // means match number of available cores
    unsigned int model_limit = 1;      // The maximum number of models extracted

    bool verbose = false;      // Flag for printing status to standard output
    bool diagnostics = false;  // Flag for printing diagnosis to standard output
                               // if a bug is detected

    unsigned char depth_budget = 0;  // The maximum tree depth for solutions, counting a tree with just
                                     // the root node as depth 1. 0 means unlimited.
    bool reference_LB = false;       // Flag for using a vector of misclassifications from another
                                     // (reference) model to lower bound our own misclassifications
    bool look_ahead = true;         // Flag for enabling the one-step look-ahead bound
                                    // implemented via scopes
    bool similar_support = true;    // Flag for enabling the similar support bound
                                    // imeplemented via the distance index
    bool cancellation = true;       // Flag for enabling upward propagation of cancelled subproblems
    bool feature_transform = true;  // Flag for enabling the equivalence discovery
                                    // through simple feature transformations
    bool rule_list = false;         // Flag for enabling rule-list constraints on models
    bool non_binary = false;        // Flag for enabling non-binary encoding
    unsigned char cart_lookahead_depth = 0;    // When nonzero, switches gosdt into a search for optimal cart prefixes of the specified depth

    std::string trace;    // Path to directory used to store traces
    std::string tree;     // Path to directory used to store tree-traces
    std::string profile;  // Path to file used to log runtime statistics

};

#endif
