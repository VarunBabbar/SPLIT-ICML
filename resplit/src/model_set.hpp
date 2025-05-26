#ifndef MODEL_SET_H
#define MODEL_SET_H

#include <map>
#include <set>
#include <string>
#include <stdlib.h>
#include <iostream>
#include <algorithm>
#include <memory>
#include <type_traits>

#include <json/json.hpp>
#include <unordered_map>

#include "configuration.hpp"
#include "graph.hpp"
#include "state.hpp"

#include "cart_it.hpp"
#include "additive_metrics.hpp"

using json = nlohmann::json;

class ModelSet;

typedef std::shared_ptr<ModelSet> model_set_p;
// typedef std::vector< std::pair< model_set_p, model_set_p> > children_set_t;




// <TP, TN, # of leaves>
// typedef std::tuple<int, int, int> values_of_interest_t;
typedef ValuesOfInterest values_of_interest_t;

struct key_hash {
    std::size_t operator()(const values_of_interest_t &k) const {
        return k.hash();
    }
};

// Count of each values of interest
typedef std::unordered_map<values_of_interest_t, int, key_hash> values_of_interest_count_t;
typedef std::unordered_map<values_of_interest_t, model_set_p, key_hash> values_of_interest_mapping_t;


enum ModelSetType {
    CLUSTERED_BY_OBJ,
    CLUSTERED_BY_TUPLE,
};

// Container for holding classification model extracted from the dependency graph
class ModelSet {
public:
    ModelSet(ModelSetType type=CLUSTERED_BY_OBJ);
    // Constructor for terminal node in a model
    // @param set: shared pointer to a bitmask that identifies the captured set of data points
    ModelSet(std::shared_ptr<Bitmask> set);

    // Constructor for terminal node for switching node type
    ModelSet(ModelSet* source);

    ~ModelSet(void);

    // Hash generated from the leaf set of model
    size_t hash(void) const;

    // Equality operator implemented by comparing the set of addresses of the bitmask of each leaf
    // @param other: other model to compare against
    // @returns true if the two models are provably equivalent
    // @note the equality comparison assumes that leaf bitmasks are not duplicated
    //       this assumes that identical bitmasks are only copy by reference, not by value
    bool operator==(ModelSet const & other) const;

    // @returns: the training loss incurred by this model
    float loss(void) const;

    // @returns: the complexity penalty incurred by this model
    float complexity(void) const;
    
    // @returns: the objective value of this model
    // float objective_value(void) const;
    
    void insert(int feature, model_set_p & positive, model_set_p & negative);
    
    void merge_with_leaf(ModelSet* other);
    void merge_with_leaf(model_set_p & other);

    void merge(model_set_p & other);

    unsigned int get_binary_target() {
        return binary_target;
    };
    
    // boost::multiprecision::uint128_t get_stored_model_count();
    long long unsigned int get_stored_model_count();
    
    // bad coding practice but whatever lol
    values_of_interest_t merge_values_of_interest_with_self(values_of_interest_t other) {
        assert(terminal);
        return values_of_interest + other;
    };
    
    values_of_interest_count_t& get_values_of_interest_count();
    values_of_interest_mapping_t& get_values_of_interest_mapping();

    void construct_values_of_interest_count();
    void construct_values_of_interest_mapping();
    
    static void serialize(results_t source, std::string & serialization, int const spacing);
    static void serialize(values_of_interest_mapping_t source, std::string & serialization, int const spacing);
    static void convert_ptr_and_to_json(model_set_p const source, json & storage_arr, std::unordered_map<ModelSet*, int> & pointer_dictionary);
    static json convert_values_of_interest_to_array(values_of_interest_t values_of_interest);

    bool terminal = false;
    const ModelSetType type;
    Objective objective;

private:

    values_of_interest_count_t values_of_interest_count;
    values_of_interest_mapping_t values_of_interest_mapping;
    

    // Non-terminal members
    std::unordered_map<int, std::vector< std::pair< model_set_p, model_set_p> >> mapping;


    // boost::multiprecision::uint128_t stored_model_count = 0;
    long long unsigned int stored_model_count = 0;

    // Terminal members
    unsigned int binary_target; // index of the encoded prediction
    float _loss; // loss incurred by this leaf
    float _complexity; // complexity penalty incurred by this leaf
    ValuesOfInterest values_of_interest;
    
    friend class Trie;
};

namespace std {
    
}

#endif