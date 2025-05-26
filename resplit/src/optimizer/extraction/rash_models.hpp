#include <memory>
void Optimizer::rash_models(results_t & results) {
    if (Configuration::model_limit == 0) { return; }
    rash_models(this -> root, results, rashomon_bound);

    if (Configuration::verbose) {
        std::cout << "Cached subproblem models size: " << State::graph.models.size() << std::endl;
        std::cout << "Models calls: " << models_calls << std::endl;
        std::cout << "Pruned combinations using scope: " << pruned_combinations_with_scope << std::endl;
        std::cout << "Pruned leaves using scope: " << pruned_leaves_with_scope << std::endl;
        std::cout << "Pruned trivial extensions: " << pruned_trivial_extension << std::endl;
        std::cout << "Max results size: " << max_result_size << std::endl;
        std::cout << "Re-explore by scope count: " << re_explore_by_scope_update_count << std::endl;
        std::cout << "Re-explore count: " << re_explore_count << std::endl;

    }
}

void Optimizer::rash_models(key_type const & identifier, results_t & results, float scope) {
    // Shortcircuit model extraction if number of models exceeds given amount
    if (model_limit_exceeded) {
        return;
    }

    models_calls++;

    models_accessor stored_models_accessor;
    if (State::graph.models.find(stored_models_accessor, identifier)) {
        scoped_result_t stored_models = stored_models_accessor->second;
        stored_models_accessor.release();
        float stored_scope = std::get<0>(stored_models);
        if (stored_scope >= scope) {
            auto& stored_models_set = std::get<1>(stored_models);
            auto& key_set = stored_models_set.first;
            // TODO: exclude out of scope stuff here
            results = stored_models_set;
            // for (std::shared_ptr<Model> model : stored_models_set) {
            //     if (model->loss() + model->complexity() <= scope) {
            //         results.insert(model);
            //     }
            // }
            return;
        } else {
            State::graph.models.erase(identifier);
        }
    } 

    rash_models_inner(identifier, results, scope);

    auto new_models = std::make_pair(identifier, std::make_pair(scope, results));
    State::graph.models.insert(new_models);
}

void Optimizer::rash_models_inner(key_type const & identifier, results_t & results, float scope) {
    vertex_accessor task_accessor;
    if (State::graph.vertices.find(task_accessor, identifier) == false) { return; }
    Task & task = task_accessor -> second;
    //std::cout << "Base Condition: " << task.base_objective() << " <= " << task.upperbound() << " = " << (int)(task.base_objective() <= task.upperbound()) << std::endl;

    // std::cout << "Capture: " << task.capture_set().to_string() << std::endl;

    if (task.maximum_scope > 0) {
        re_explore_count++;
    }

    if (task.maximum_scope < scope) {
        if (task.maximum_scope > 0) {
            re_explore_by_scope_update_count++;
        }
        task.maximum_scope = scope;
    }
    
    auto& keys = results.first;
    auto& storage = results.second;

    if (task.base_objective() <= scope + std::numeric_limits<float>::epsilon()) {
        // || (Configuration::rule_list && task.capture_set().count() != task.capture_set().size())) {
        // std::cout << "Stump" << std::endl;
        // std::shared_ptr<key_type> stump(new Tile(set));
        // Model stump_key(stump_set); // shallow variant
        // Model * stump_address = new Model(stump_set);
        //std::cout << task.rashomon_bound() << std::endl;


        model_set_p model(new ModelSet(std::shared_ptr<Bitmask>(new Bitmask(task.capture_set()))));
        insert_leaf_to_results(results, model);        
        // std::shared_ptr<Model> model(new Model(std::shared_ptr<Bitmask>(new Bitmask(task.capture_set()))));
        // model -> identify(identifier);
        // model -> translate_self(task.order());
        // results.insert(model);
    } else {
        pruned_leaves_with_scope++;
    }

    bound_accessor bounds;
    float lower_val, upper_val;
    if (!State::graph.bounds.find(bounds, identifier)) { return; }

    
    for (bound_iterator iterator = bounds -> second.begin(); iterator != bounds -> second.end(); ++iterator) {

        if (std::get<1>(* iterator) > scope + std::numeric_limits<float>::epsilon()) { continue; }
        int feature = std::get<0>(* iterator);
        //std::cout << "Feature: " << feature << std::endl;
        results_t negatives;
        results_t positives;
        // std::unordered_set< std::shared_ptr<Model> > negatives;
        // std::unordered_set< std::shared_ptr<Model> > positives;
        bool ready = true;

        child_accessor left_key, right_key;
        vertex_accessor left_child, right_child;
        Tile left_identifier, right_identifier;
        float left_lowerbound = 0, right_lowerbound = 0;

        bool left_has_key = State::graph.children.find(left_key, std::make_pair(identifier, -(feature + 1)));
        if (left_has_key) {
            left_identifier = left_key->second;
            left_key.release();
        } else {
            Bitmask subset(task.capture_set());
            State::dataset.subset(feature, false, subset);
            // One optimization: move subset data to tile instead of copying? 
            left_identifier = Tile(subset, 0);
        }
        bool left_has_child = State::graph.vertices.find(left_child, left_identifier);
        if (left_has_child) {
            left_lowerbound = left_child->second.lowerbound();
            left_child.release();
        } else if (!left_has_key) {
            Bitmask &subset = left_identifier.content();
            model_set_p model(new ModelSet(std::shared_ptr<Bitmask>(new Bitmask(subset))));
            float leaf_objective = model->loss() + model->complexity();
            if (leaf_objective > scope + std::numeric_limits<float>::epsilon()) {
                pruned_leaves_with_scope++;
                continue;
            }
            left_lowerbound = leaf_objective;
            insert_leaf_to_results(negatives, model);        
        } else {
            continue;
        }

        bool right_has_key = State::graph.children.find(right_key, std::make_pair(identifier, feature + 1));
        if (right_has_key) {
            right_identifier = right_key->second;
            right_key.release();
        } else {
            Bitmask subset(task.capture_set());
            State::dataset.subset(feature, true, subset);
            right_identifier = Tile(subset, 0);
        }
        bool right_has_child = State::graph.vertices.find(right_child, right_identifier);
        if (right_has_child) {
            right_lowerbound = right_child->second.lowerbound();
            right_child.release();
        } else if (!right_has_key) {
            Bitmask &subset = right_identifier.content();
            unsigned int count = subset.count();
            model_set_p model(new ModelSet(std::shared_ptr<Bitmask>(new Bitmask(subset))));
            float leaf_objective = model->loss() + model->complexity();
            if (leaf_objective > scope + std::numeric_limits<float>::epsilon()) {
                pruned_leaves_with_scope++;
                continue;
            }
            right_lowerbound = leaf_objective;
            insert_leaf_to_results(positives, model);        
        } else {
            // might never reach here? 
            continue;
        }

        if ((scope - right_lowerbound < 0 || scope - left_lowerbound < 0)) { continue; }


        float left_scope = scope - right_lowerbound;
        if (left_has_child) {    
            rash_models(left_identifier, negatives, left_scope);

            // std::unordered_set< std::shared_ptr<Model> > negatives_old;
            // models(left_key -> second, negatives_old, left_scope);
            // unsigned long models_count = 0;
            // for (auto key : negatives.first) {
            //     if (key > left_scope) break;
            //     models_count += negatives.second.find(key)->second->get_stored_model_count();
            // }
            // assert(negatives_old.size() == models_count);

        }

        if (negatives.first.size() == 0 || *negatives.first.begin() > left_scope) { continue; }

        float right_scope = scope - left_lowerbound;
        if (right_has_child) {
            rash_models(right_identifier, positives, right_scope);
        } 

        if (positives.first.size() == 0 || *positives.first.begin() > right_scope) { continue; }
        
        {
            // results_t& negatives = negatives;
            // const results_t& positives = positives;
            if (Configuration::rule_list) {
                throw std::invalid_argument("Does not support rule lists");
            } else {

                for (auto negative_it = negatives.first.begin(); negative_it != negatives.first.end() && *negative_it <= left_scope; ++negative_it) {
                    for (auto positive_it = positives.first.begin(); positive_it != positives.first.end() && *positive_it <= right_scope; ++positive_it) {
                    // for (auto positive_it = positives.begin(); positive_it != positives.end(); ++positive_it) {

                        if (Configuration::model_limit > 0 && results.first.size() > Configuration::model_limit) { 
                            model_limit_exceeded = true;
                            return;
                        }

                        Objective new_key = *negative_it + *positive_it;
                        // Prune models exceeding maximum allowed objective value 
                        if (new_key > scope) {
                            pruned_combinations_with_scope++; 
                            break; 
                        }

                        auto& negative_model = negatives.second.find(*negative_it)->second;
                        auto& positive_model = positives.second.find(*positive_it)->second;
                        // std::shared_ptr<Model> negative_model = (* negative_it);
                        // std::shared_ptr<Model> positive_model = (* positive_it);

                        // Prune trivial extensions
                        if (Configuration::rashomon_ignore_trivial_extensions 
                                && negative_model->terminal == positive_model->terminal 
                                && (negative_model->terminal 
                                    ? negative_model->get_binary_target() == positive_model->get_binary_target()
                                    : false)) {
                            pruned_trivial_extension++;
                            continue;
                        }

                        
                        auto iter = storage.find(new_key);
                        if (iter == storage.end()) {
                            keys.insert(new_key);
                            iter = storage.insert(std::pair<Objective, model_set_p>(new_key, std::make_shared<ModelSet>())).first;
                        }
                        iter->second->insert(feature, positive_model, negative_model);
                    }
                }

            }
        }
    }

    max_result_size = std::max(max_result_size, keys.size());
    return;
}


// Must be constructing the results! (i.e. cannot operate on constructed negatives etc.)
void Optimizer::insert_leaf_to_results(results_t & results, model_set_p & model) {
    auto key = model->objective;
    
    auto iter = results.second.find(key);
    if (iter == results.second.end()) {
        results.second.emplace(std::pair<Objective, model_set_p>(key, model));
        results.first.emplace(key);
    } else {
        iter->second->merge_with_leaf(model);
    }
}

