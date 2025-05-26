#pragma once

#include <algorithm>
#include <array>
#include <iterator>
#include <map>
#include <memory>
#include <stdlib.h>
#include <typeinfo>
#include <vector>

#include "model.hpp"
#include "model_set.hpp"

#include "cart_it.hpp"

// #include "utils.h"
// #include "alloc.h"
// #include "rule.h"

enum class DataStruct { Tree, Queue, Pmap };
template <class T, DataStruct S>
using tracking_vector = std::vector<T, std::allocator<T>>;
typedef struct rule {
    char *features; /* Representation of the rule. */
    int support;    /* Number of 1's in truth table. */
    int cardinality;
    int *ids;
    // VECTOR truthtable;		/* Truth table; one bit per sample. */
} rule_t;

class Node {
  public:
    Node();
    Node(std::vector<int> id, Node *parent);

    virtual ~Node();

    inline std::vector<int> id() const;

    // Returns pair of prefixes and predictions for the path from this node to
    // the root
    inline std::pair<tracking_vector<std::vector<int>, DataStruct::Tree>,
                     tracking_vector<bool, DataStruct::Tree>>
    get_prefix_and_predictions();

    inline size_t depth() const;
    inline Node *child(std::vector<int> idx);
    inline Node *parent() const;
    inline void delete_child(std::vector<int> idx);
    inline size_t num_children() const;

    inline typename std::map<std::vector<int>, Node *>::iterator
    children_begin();
    inline typename std::map<std::vector<int>, Node *>::iterator children_end();
    virtual inline double get_curiosity() { return 0.0; }

    bool terminal = false; // Flag specifying whether the node is terminal

    // @modifies node: JSON object representation of this model
    void to_json(json &node) const;

  protected:
    std::map<std::vector<int>, Node *> children_;

    Node *parent_;
    size_t depth_;
    std::vector<int> id_;

    // Terminal members
    float loss;
    float complexity;

    friend class Trie;
};

class Trie {
  public:
    Trie(){};
    Trie(bool calculate_size, char const *type);
    ~Trie();

    Node *construct_node(std::vector<int> new_rule, Node *parent);

    inline size_t num_nodes() const;
    inline size_t num_evaluated() const;
    inline Node *root() const;

    void insert_model(const Model *model);

    void insert_model_set(const model_set_p model);
    // @param models: a vector of children sets. Any element in the cartesian
    // product of the sets is in the rashomon set and (supposedly) has the same
    // objective value by design
    void insert_model_set_children(
        const std::vector<std::vector<std::pair<model_set_p, model_set_p>>>
            &models,
        Node *currNode, values_of_interest_t values_of_interest);

    void finalize_leaf_node(Node *currNode,
                            values_of_interest_t values_of_interest);

    inline void increment_num_evaluated();
    inline void decrement_num_nodes();
    inline int ablation() const;
    inline bool calculate_size() const;

    void insert_root();
    void insert(Node *node);
    void insert_if_not_exist(std::vector<int> feats, Node *currNode,
                             Node *&child);
    void prune_up(Node *node);
    Node *
    check_prefix(tracking_vector<std::vector<int>, DataStruct::Tree> &prefix);

    // @param spacing: number of spaces to used in the indentation format
    // @modifies serialization: string representation of the JSON object
    // representation of this model
    void serialize(std::string &serialization, int const spacing = 0) const;

  protected:
    Node *root_;

    size_t num_nodes_;
    size_t num_evaluated_;
    bool calculate_size_;

    char const *type_;
    void gc_helper(Node *node);
};

inline std::vector<int> Node::id() const { return id_; }

// A function for debugging.
// inline std::pair<tracking_vector<std::vector<int>, DataStruct::Tree>,
// tracking_vector<bool, DataStruct::Tree> >
//     Node::get_prefix_and_predictions() {
//     tracking_vector<std::vector<int>, DataStruct::Tree> prefix;
//     tracking_vector<bool, DataStruct::Tree> predictions;
//     tracking_vector<std::vector<int>, DataStruct::Tree>::iterator it1 =
//     prefix.begin(); tracking_vector<bool, DataStruct::Tree>::iterator it2 =
//     predictions.begin(); Node* node = this; for(size_t i = depth_; i > 0;
//     --i) {
//         it1 = prefix.insert(it1, node->id());
//         it2 = predictions.insert(it2, node->prediction());
//         node = node->parent();
//     }
//     return std::make_pair(prefix, predictions);
// }

inline size_t Node::depth() const { return depth_; }

inline Node *Node::child(std::vector<int> idx) {
    typename std::map<std::vector<int>, Node *>::iterator iter;
    iter = children_.find(idx);
    if (iter == children_.end())
        return NULL;
    else
        return iter->second;
}

inline void Node::delete_child(std::vector<int> idx) { children_.erase(idx); }

inline size_t Node::num_children() const { return children_.size(); }

inline typename std::map<std::vector<int>, Node *>::iterator
Node::children_begin() {
    return children_.begin();
}

inline typename std::map<std::vector<int>, Node *>::iterator
Node::children_end() {
    return children_.end();
}

inline Node *Node::parent() const { return parent_; }

inline size_t Trie::num_nodes() const { return num_nodes_; }

inline size_t Trie::num_evaluated() const { return num_evaluated_; }

inline Node *Trie::root() const { return root_; }

inline bool Trie::calculate_size() const { return calculate_size_; }

/*
 * Increment number of nodes evaluated after performing incremental computation
 * in evaluate_children.
 */
inline void Trie::increment_num_evaluated() { ++num_evaluated_; }

/*
 * Called whenever a node is deleted from the tree.
 */
inline void Trie::decrement_num_nodes() { --num_nodes_; }

void delete_subtree(Trie *tree, Node *node, bool destructive,
                    bool update_remaining_state_space);
