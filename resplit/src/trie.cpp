#include "trie.hpp"
#include "configuration.hpp"
#include <array>
#include <iostream>
#include <memory>
#include <stdlib.h>
#include <string>
#include <vector>

#include <sstream>

Node::Node() : depth_(0), id_(std::vector<int>{}) {}

Node::~Node() {}

Node::Node(std::vector<int> id, Node *parent)
    : parent_(parent), depth_(1 + parent->depth_), id_(id) {}

Trie::Trie(bool calculate_size, char const *type)
    : root_(0), num_nodes_(0), num_evaluated_(0),
      calculate_size_(calculate_size), type_(type) {}

Trie::~Trie() {
    // if(num_nodes())
    //     delete_subtree(this, root_, true, false);
}

Node *Trie::construct_node(std::vector<int> new_rule, Node *parent) {
    Node *n;
    n = (new Node(new_rule, parent));
    return n;
}

/*
 * Inserts the root of the tree, setting up the default rules.
 */
void Trie::insert_root() {
    root_ = new Node();
    ++num_nodes_;
}

/*
 * Insert a node into the tree.
 */
void Trie::insert(Node *node) {
    node->parent()->children_.insert(std::make_pair(node->id(), node));
    ++num_nodes_;
}

void Trie::insert_if_not_exist(std::vector<int> feats, Node *currNode,
                               Node *&child) {
    child = currNode->child(feats);
    if (!child) {
        auto newNode = construct_node(feats, currNode);
        insert(newNode);
        child = newNode;
    }
}

void Trie::insert_model(const Model *model) {
    std::vector<const Model *> models{model};
    Node *currNode = root_;
    while (!models.empty()) {
        std::vector<int> feats;
        std::vector<const Model *> newModels;

        for (const Model *m : models) {
            unsigned int feat;
            if (m->terminal) {
                feat = -m->get_binary_target() - 1;
            } else {
                feat = m->get_feature();
                newModels.emplace_back(m->get_positive().get());
                newModels.emplace_back(m->get_negative().get());
            }
            feats.emplace_back(feat);
        }
        if (!currNode->child(feats)) {
            auto newNode = construct_node(feats, currNode);
            insert(newNode);
        }
        currNode = currNode->child(feats);
        models = newModels;
    }
    currNode->terminal = true;
    currNode->loss = model->loss();
    currNode->complexity = model->complexity();
}

void Trie::insert_model_set(const model_set_p models) {
    if (models->terminal) {
        Node *child;
        std::vector<int> feats = {-(int)models->get_binary_target() - 1};
        insert_if_not_exist(feats, root_, child);
        finalize_leaf_node(child, models->values_of_interest);
    }
    for (auto it : models->mapping) {
        Node *child;
        std::vector<int> feats = {it.first};
        insert_if_not_exist(feats, root_, child);
        std::vector<std::vector<std::pair<model_set_p, model_set_p>>>
            init_combinations = {it.second};
        insert_model_set_children(init_combinations, child,
                                  values_of_interest_t());
    }
}

void Trie::insert_model_set_children(
    const std::vector<std::vector<std::pair<model_set_p, model_set_p>>> &models,
    Node *currNode, values_of_interest_t values_of_interest) {
    CartIt<std::pair<model_set_p, model_set_p>> cart_container(models);

    for (auto selected_extensions : cart_container) {
        // We must have negative and positive paired together at all times so
        // we generate cartesian product on the pair rather than loose
        // extensions. However, when flattening to one layer in the trie, we
        // must unwarp them
        std::vector<model_set_p> unwrapped_models;
        for (auto selected_extension : selected_extensions) {
            unwrapped_models.emplace_back(selected_extension.first);
            unwrapped_models.emplace_back(selected_extension.second);
        }

        // Gather all available features for generating the cartesian product
        // of features available in each model set container
        std::vector<std::vector<int>> next_trie_level_keys;
        for (auto unwrapped_model : unwrapped_models) {
            std::vector<int> available_features;

            for (auto feature_value_pair : unwrapped_model->mapping) {
                available_features.emplace_back(feature_value_pair.first);
            }
            if (unwrapped_model->terminal) {
                available_features.emplace_back(
                    -(int)unwrapped_model->get_binary_target() - 1);
            }
            next_trie_level_keys.push_back(available_features);
        }

        CartIt<int> feature_cart_container(next_trie_level_keys);

        for (auto next_trie_level_key : feature_cart_container) {
            Node *child;
            values_of_interest_t new_values_of_interest = values_of_interest;
            std::vector<std::vector<std::pair<model_set_p, model_set_p>>>
                next_level_models;
            for (int i = 0; i < next_trie_level_key.size(); i++) {
                if (next_trie_level_key[i] >= 0) {
                    next_level_models.push_back(
                        unwrapped_models[i]->mapping[next_trie_level_key[i]]);
                } else {
                    new_values_of_interest =
                        unwrapped_models[i]->merge_values_of_interest_with_self(
                            new_values_of_interest);
                }
            }
            insert_if_not_exist(next_trie_level_key, currNode, child);
            if (next_level_models.size() == 0) {
                finalize_leaf_node(child, new_values_of_interest);
            } else {
                insert_model_set_children(next_level_models, child,
                                          new_values_of_interest);
            }
        }
    }
}

void Trie::finalize_leaf_node(Node *currNode,
                              values_of_interest_t values_of_interest) {
    currNode->terminal = true;
    double TP = values_of_interest.TP;
    double TN = values_of_interest.TN;
    auto reg = values_of_interest.regularization;

    currNode->loss = 1 - (TP + TN) / (float)State::dataset.size();
    currNode->complexity = reg * Configuration::regularization;
}

void Node::to_json(json &node) const {
    if (this->terminal) {
        node["loss"] =
            this->loss; // This value is correct regardless of translation
        node["complexity"] = this->complexity;
        node["objective"] = this->loss + this->complexity;

    } else {
        for (auto i : this->children_) {
            auto v = i.first;
            json child = json::object();

            std::stringstream ss;
            copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
            std::string s = ss.str();
            s = s.substr(0, s.length() - 1); // get rid of the trailing space

            i.second->to_json(child);
            node[s] = child;
        }
    }
    return;
}

void Trie::serialize(std::string &serialization, int const spacing) const {
    json node = json::object();
    root_->to_json(node);
    serialization = spacing == 0 ? node.dump() : node.dump(spacing);
    return;
}

/*
 * Removes nodes with no children, recursively traversing tree towards the root.
 */
void Trie::prune_up(Node *node) {
    std::vector<int> id;
    size_t depth = node->depth();
    Node *parent;
    while (node->children_.size() == 0) {
        if (depth > 0) {
            id = node->id();
            parent = node->parent();
            parent->children_.erase(id);
            --num_nodes_;
            delete node;
            node = parent;
            --depth;
        } else {
            --num_nodes_;
            break;
        }
    }
}

/*
 * Checks that the prefix is in the tree and hasn't been deleted.
 * Returns NULL if the prefix isn't in the tree, a pointer to the prefix node
 * otherwise.
 */
Node *Trie::check_prefix(
    tracking_vector<std::vector<int>, DataStruct::Tree> &prefix) {
    Node *node = this->root_;
    for (tracking_vector<std::vector<int>, DataStruct::Tree>::iterator it =
             prefix.begin();
         it != prefix.end(); ++it) {
        node = node->child(*it);
        if (node == NULL)
            return NULL;
    }
    return node;
}
