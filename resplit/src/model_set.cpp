#include "model_set.hpp"
#include "graph.hpp"
#include <array>
#include <cassert>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

ModelSet::ModelSet(ModelSetType type) : type(type) {}

ModelSet::ModelSet(std::shared_ptr<Bitmask> capture_set)
    : type(CLUSTERED_BY_OBJ) {
    std::string prediction_name, prediction_type, prediction_value;
    float info, potential, min_loss, max_loss;
    unsigned int target_index;
    // TODO: investigate performance impact of the following line
    State::dataset.summary(*capture_set, info, potential, min_loss, max_loss,
                           target_index, 0);
    State::dataset.encoder.target_value(target_index, prediction_value);
    // State::dataset.encoder.header(prediction_name);
    // State::dataset.encoder.target_type(prediction_type);
    unsigned int TP, TN;
    State::dataset.get_TP_TN(*capture_set, 0, target_index, TP, TN);

    this->binary_target = target_index;
    // this -> name = prediction_name;
    // this -> type = prediction_type;
    // this -> prediction = prediction_value;
    this->_loss = max_loss;
    this->_complexity = Configuration::regularization;
    // this -> capture_set = capture_set;
    this->terminal = true;
    this->objective = Objective(capture_set->count() - TP - TN, 1);
    this->values_of_interest = ValuesOfInterest(TP, TN, 1);
    stored_model_count++;
}

ModelSet::ModelSet(ModelSet *source) : type(CLUSTERED_BY_TUPLE) {
    assert(source->type == CLUSTERED_BY_OBJ);
    merge_with_leaf(source);
}

ModelSet::~ModelSet(void) {}

float ModelSet::loss(void) const {
    assert(terminal);
    return _loss;
}

float ModelSet::complexity(void) const {
    assert(terminal);
    return _complexity;
}

// boost::multiprecision::uint128_t ModelSet::get_stored_model_count() {
long long unsigned int ModelSet::get_stored_model_count() {
    // unsigned long out = 0;
    // if (terminal) out++;

    // for (auto it : mapping) {
    //     for (auto pair : it.second) {
    //         out += pair.first->get_stored_model_count() *
    //         pair.second->get_stored_model_count();
    //     }
    // }
    // assert(stored_model_count == out);
    // return out;

    return stored_model_count;
}

void ModelSet::insert(int feature, model_set_p &positive,
                      model_set_p &negative) {
    assert(type == positive->type && type == negative->type);
    auto iter = mapping.find(feature);
    if (iter == mapping.end()) {
        iter = mapping
                   .insert(std::make_pair(
                       feature,
                       std::vector<std::pair<model_set_p, model_set_p>>()))
                   .first;
    }
    auto &vector = iter->second;
    vector.emplace_back(std::make_pair(positive, negative));
    stored_model_count +=
        positive->get_stored_model_count() * negative->get_stored_model_count();
}

void ModelSet::merge_with_leaf(model_set_p &other) {
    // Bad practice? But it should terminate immediately...
    merge_with_leaf(other.get());
}

void ModelSet::merge(model_set_p &other) {
    assert(type == other->type);
    // this relies on having mapping -> is not a leaf
    assert(!terminal && !other->terminal);
    for (auto other_feature : other->mapping) {
        auto local_feature = mapping.find(other_feature.first);
        if (local_feature == mapping.end()) {
            mapping.insert(other_feature);
        } else {
            local_feature->second.insert(local_feature->second.end(),
                                         other_feature.second.begin(),
                                         other_feature.second.end());
        }
    }
    stored_model_count += other->stored_model_count;
}

void ModelSet::merge_with_leaf(ModelSet *other) {
    assert(!terminal && other->terminal);
    stored_model_count++;
    terminal = other->terminal;
    binary_target = other->binary_target;
    _loss = other->_loss;
    _complexity = other->_complexity;
    values_of_interest = other->values_of_interest;
}

values_of_interest_count_t &ModelSet::get_values_of_interest_count() {
    construct_values_of_interest_count();
    return values_of_interest_count;
}

void ModelSet::construct_values_of_interest_count() {
    if (values_of_interest_count.size() > 0)
        return;

    if (terminal)
        values_of_interest_count.insert(std::make_pair(values_of_interest, 1));

    for (auto feature : mapping) {
        for (auto pair : feature.second) {
            std::vector<std::pair<values_of_interest_t, int>> left, right;
            for (auto i : pair.first->get_values_of_interest_count()) {
                left.push_back(i);
            }
            for (auto i : pair.second->get_values_of_interest_count()) {
                right.push_back(i);
            }
            std::vector<std::vector<std::pair<values_of_interest_t, int>>>
                pairs({left, right});

            CartIt<std::pair<values_of_interest_t, int>> cart_container_pairs(
                pairs);

            for (auto i : cart_container_pairs) {
                values_of_interest_t left_values_of_interest = i[0].first;
                values_of_interest_t right_values_of_interest = i[1].first;

                values_of_interest_t new_values_of_interest =
                    left_values_of_interest + right_values_of_interest;

                auto it = values_of_interest_count.find(new_values_of_interest);
                if (it == values_of_interest_count.end()) {
                    it = values_of_interest_count
                             .insert(std::make_pair(new_values_of_interest, 0))
                             .first;
                }
                it->second += i[0].second * i[1].second;
            }
        }
    }
}

values_of_interest_mapping_t &ModelSet::get_values_of_interest_mapping() {
    construct_values_of_interest_mapping();
    return values_of_interest_mapping;
}

void ModelSet::construct_values_of_interest_mapping() {

    if (values_of_interest_mapping.size() > 0)
        return;

    if (terminal) {
        model_set_p new_leaf(new ModelSet(this));
        // Must be unique here
        values_of_interest_mapping.insert(
            std::make_pair(values_of_interest, new_leaf));
    }

    for (auto feature : mapping) {
        for (auto pair : feature.second) {
            std::vector<std::pair<values_of_interest_t, model_set_p>> left,
                right;
            for (auto i : pair.first->get_values_of_interest_mapping()) {
                left.push_back(i);
            }
            for (auto i : pair.second->get_values_of_interest_mapping()) {
                right.push_back(i);
            }
            std::vector<
                std::vector<std::pair<values_of_interest_t, model_set_p>>>
                pairs({left, right});

            CartIt<std::pair<values_of_interest_t, model_set_p>>
                cart_container_pairs(pairs);

            for (auto i : cart_container_pairs) {
                values_of_interest_t left_values_of_interest = i[0].first;
                values_of_interest_t right_values_of_interest = i[1].first;

                values_of_interest_t new_values_of_interest =
                    left_values_of_interest + right_values_of_interest;

                auto it =
                    values_of_interest_mapping.find(new_values_of_interest);
                if (it == values_of_interest_mapping.end()) {
                    model_set_p new_container(new ModelSet(CLUSTERED_BY_TUPLE));
                    it = values_of_interest_mapping
                             .insert(std::make_pair(new_values_of_interest,
                                                    new_container))
                             .first;
                }
                it->second->insert(feature.first, i[0].second, i[1].second);
            }
        }
    }
}

// TODO: merge this function with the one below
void ModelSet::serialize(results_t source, std::string &serialization,
                         int const spacing) {
    std::unordered_map<ModelSet *, int> pointer_dictionary;
    json node = json::object();
    json metric_values = json::array();
    json metric_pointers = json::array();
    node["storage"] = json::array();
    for (auto i : source.second) {
        pointer_dictionary[i.second.get()] = pointer_dictionary.size();

        convert_ptr_and_to_json(i.second, node["storage"], pointer_dictionary);
        metric_values.emplace_back(i.first.to_tuple());
        metric_pointers.emplace_back(pointer_dictionary[i.second.get()]);
    }
    if (Configuration::verbose) {
        std::cout << "Nodes Count in Model Set: " << pointer_dictionary.size()
                  << std::endl;
    }
    node["available_metric_values"] = json::object();
    node["available_metric_values"]["metric_values"] = metric_values;
    node["available_metric_values"]["metric_pointers"] = metric_pointers;

    node["metadata"] = json::object();
    node["metadata"]["regularization"] = Configuration::regularization;
    node["metadata"]["dataset_size"] = State::dataset.size();
    serialization = spacing == 0 ? node.dump() : node.dump(spacing);
}

void ModelSet::serialize(values_of_interest_mapping_t source,
                         std::string &serialization, int const spacing) {
    std::unordered_map<ModelSet *, int> pointer_dictionary;
    json node = json::object();
    json metric_values = json::array();
    json metric_pointers = json::array();
    node["storage"] = json::array();
    for (auto i : source) {
        pointer_dictionary[i.second.get()] = pointer_dictionary.size();

        convert_ptr_and_to_json(i.second, node["storage"], pointer_dictionary);
        metric_values.emplace_back(i.first.to_tuple());
        metric_pointers.emplace_back(pointer_dictionary[i.second.get()]);
    }
    if (Configuration::verbose) {
        std::cout << "Nodes Count in Model Set: " << pointer_dictionary.size()
                  << std::endl;
    }
    node["available_metric_values"] = json::object();
    node["available_metric_values"]["metric_values"] = metric_values;
    node["available_metric_values"]["metric_pointers"] = metric_pointers;

    node["metadata"] = json::object();
    node["metadata"]["regularization"] = Configuration::regularization;
    node["metadata"]["dataset_size"] = State::dataset.size();
    serialization = spacing == 0 ? node.dump() : node.dump(spacing);
}

void ModelSet::convert_ptr_and_to_json(
    model_set_p const source, json &storage_arr,
    std::unordered_map<ModelSet *, int> &pointer_dictionary) {

    json node = json::object();

    // node["values_of_interest"] =
    // convert_values_of_interest_to_array(source->values_of_interest);

    json mapping = json::object();

    for (auto i : source->mapping) {
        json feature_extensions = json::array();
        for (auto pair : i.second) {
            ModelSet *left_ptr = pair.first.get();
            auto left_ptr_conv = pointer_dictionary.find(left_ptr);
            if (left_ptr_conv == pointer_dictionary.end()) {
                left_ptr_conv = pointer_dictionary
                                    .insert(std::make_pair(
                                        left_ptr, pointer_dictionary.size()))
                                    .first;
                convert_ptr_and_to_json(pair.first, storage_arr,
                                        pointer_dictionary);
            }
            ModelSet *right_ptr = pair.second.get();
            auto right_ptr_conv = pointer_dictionary.find(right_ptr);
            if (right_ptr_conv == pointer_dictionary.end()) {
                right_ptr_conv = pointer_dictionary
                                     .insert(std::make_pair(
                                         right_ptr, pointer_dictionary.size()))
                                     .first;
                convert_ptr_and_to_json(pair.second, storage_arr,
                                        pointer_dictionary);
            }

            // json pair_json = json::array();
            // pair_json.emplace_back(*left_ptr_conv);
            // pair_json.emplace_back(*right_ptr_conv);

            feature_extensions.emplace_back(
                json::array({left_ptr_conv->second, right_ptr_conv->second}));
        }
        unsigned int binary_feature_index = i.first;
        unsigned int feature_index;
        State::dataset.encoder.decode(binary_feature_index, &feature_index);
        mapping[std::to_string(feature_index)] = feature_extensions;
    }
    // node["values_of_interest"] = source->values_of_interest;
    node["terminal"] = source->terminal;
    node["count"] = (unsigned long long)source->stored_model_count;
    node["prediction"] = source->binary_target;
    node["mapping"] = mapping;
    storage_arr[pointer_dictionary[source.get()]] = node;
}

json ModelSet::convert_values_of_interest_to_array(
    values_of_interest_t values_of_interest) {
    json::array_t out;
    out.push_back(values_of_interest.TP);
    out.push_back(values_of_interest.TN);
    out.push_back(values_of_interest.regularization);
    return out;
}
