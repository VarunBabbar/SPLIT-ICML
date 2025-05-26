#include "optimizer.hpp"

void Optimizer::models(std::unordered_set<Model> &results) {
    if (m_config.model_limit == 0) {
        return;
    }
    std::unordered_set<Model *, std::hash<Model *>, std::equal_to<Model *>> local_results;
    models(this->root, local_results);
    // Copy into final results
    for (auto iterator = local_results.begin(); iterator != local_results.end(); ++iterator) {
        results.insert(**iterator);
    }
}

void Optimizer::models(key_type const &identifier,
                       std::unordered_set<Model *, std::hash<Model *>, std::equal_to<Model *>> &results, bool leaf) {
    vertex_accessor task_accessor;
    if (m_graph.vertices.find(task_accessor, identifier) == false) {
        return;
    }
    Task &task = task_accessor->second;

    if (task.base_objective() <= task.upperbound() + std::numeric_limits<float>::epsilon()) {
        Model *model = new Model(std::shared_ptr<Bitmask>(new Bitmask(task.capture_set())), m_dataset,
                                 m_local_states[0].column_buffer);
        model->identify(identifier);

        model->translate_self(task.order());
        results.insert(model);
    }

    bound_accessor bounds;
    if (!m_graph.bounds.find(bounds, identifier)) {
        return;
    }

    for (bound_iterator iterator = bounds->second.begin(); iterator != bounds->second.end(); ++iterator) {

        if (std::get<2>(*iterator) > task.upperbound() + std::numeric_limits<float>::epsilon()) {
            continue;
        }
        int feature = std::get<0>(*iterator);
        std::unordered_set<Model *> negatives;
        std::unordered_set<Model *> positives;
        bool ready = true;

        child_accessor left_key, right_key;
        if (m_graph.children.find(left_key, std::make_pair(identifier, -(feature + 1)))) {
            models(left_key->second, negatives);
            left_key.release();
        } else {
            Bitmask subset(task.capture_set());
            m_dataset.subset_inplace(subset, feature, false);
            if (m_config.depth_budget != 0) {
                // In the case where we have a set depth_budget, children have
                // one less than their parents.
                subset.set_depth_budget(subset.get_depth_budget() - 1);
            }

            Model *model =
                new Model(std::shared_ptr<Bitmask>(new Bitmask(subset)), m_dataset, m_local_states[0].column_buffer);
            negatives.insert(model);
        }
        if (m_graph.children.find(right_key, std::make_pair(identifier, feature + 1))) {
            models(right_key->second, positives);
            right_key.release();
        } else {
            Bitmask subset(task.capture_set());
            m_dataset.subset_inplace(subset, feature, true);
            if (m_config.depth_budget != 0) {
                // In the case where we have a set depth_budget, children have
                // one less than their parents.
                subset.set_depth_budget(subset.get_depth_budget() - 1);
            }
            Model *model =
                new Model(std::shared_ptr<Bitmask>(new Bitmask(subset)), m_dataset, m_local_states[0].column_buffer);
            positives.insert(model);
        }

        if (negatives.size() == 0 || positives.size() == 0) {
            continue;
        }

        if (m_config.rule_list) {
            Bitmask negative_subset(m_dataset.m_number_rows);
            Bitmask positive_subset(m_dataset.m_number_rows);

            // Left leaf
            negative_subset = task.capture_set();
            m_dataset.subset_inplace(negative_subset, feature, false);
            if (m_config.depth_budget != 0) {
                // In the case where we have a set depth_budget, children have
                // one less than their parents.
                negative_subset.set_depth_budget(negative_subset.get_depth_budget() - 1);
            }
            Dataset::SummaryStatistics stats =
                m_dataset.summary_statistics(negative_subset, m_local_states[0].column_buffer);
            float left_leaf_risk = stats.max_loss + m_config.regularization;

            // Right leaf
            positive_subset = task.capture_set();
            m_dataset.subset_inplace(positive_subset, feature, true);
            if (m_config.depth_budget != 0) {
                // In the case where we have a set depth_budget, children have
                // one less than their parents.
                positive_subset.set_depth_budget(positive_subset.get_depth_budget() - 1);
            }
            stats = m_dataset.summary_statistics(positive_subset, m_local_states[0].column_buffer);
            float right_leaf_risk = stats.max_loss + m_config.regularization;

            for (auto negative_it = negatives.begin(); negative_it != negatives.end(); ++negative_it) {
                float risk = right_leaf_risk + (**negative_it).loss() + (**negative_it).complexity();
                if (risk <= task.upperbound() + std::numeric_limits<float>::epsilon()) {
                    if (m_config.model_limit > 0 && results.size() >= m_config.model_limit) {
                        continue;
                    }

                    std::shared_ptr<Model> negative(*negative_it);
                    std::shared_ptr<Model> positive(new Model(std::shared_ptr<Bitmask>(new Bitmask(positive_subset)),
                                                              m_dataset, m_local_states[0].column_buffer));

                    Model *model = new Model(feature, negative, positive, m_dataset);
                    model->identify(identifier);
                    model->translate_self(task.order());
                    translation_accessor negative_translation, positive_translation;
                    if (negative->identified() &&
                        m_graph.translations.find(negative_translation, std::make_pair(identifier, -(feature + 1)))) {
                        model->translate_negatives(negative_translation->second);
                    }
                    negative_translation.release();
                    if (positive->identified() &&
                        m_graph.translations.find(positive_translation, std::make_pair(identifier, feature + 1))) {
                        model->translate_positives(positive_translation->second);
                    }
                    positive_translation.release();
                    results.insert(model);
                }
            }
            for (auto positive_it = positives.begin(); positive_it != positives.end(); ++positive_it) {
                float risk = left_leaf_risk + (**positive_it).loss() + (**positive_it).complexity();
                if (risk <= task.upperbound() + std::numeric_limits<float>::epsilon()) {
                    if (m_config.model_limit > 0 && results.size() >= m_config.model_limit) {
                        continue;
                    }

                    std::shared_ptr<Model> negative(new Model(std::shared_ptr<Bitmask>(new Bitmask(negative_subset)),
                                                              m_dataset, m_local_states[0].column_buffer));
                    std::shared_ptr<Model> positive(*positive_it);

                    Model *model = new Model(feature, negative, positive, m_dataset);
                    model->identify(identifier);
                    model->translate_self(task.order());
                    translation_accessor negative_translation, positive_translation;
                    if (negative->identified() &&
                        m_graph.translations.find(negative_translation, std::make_pair(identifier, -(feature + 1)))) {
                        model->translate_negatives(negative_translation->second);
                    }
                    negative_translation.release();
                    if (positive->identified() &&
                        m_graph.translations.find(positive_translation, std::make_pair(identifier, feature + 1))) {
                        model->translate_positives(positive_translation->second);
                    }
                    positive_translation.release();
                    results.insert(model);
                }
            }
        } else {
            for (auto negative_it = negatives.begin(); negative_it != negatives.end(); ++negative_it) {
                for (auto positive_it = positives.begin(); positive_it != positives.end(); ++positive_it) {
                    if (m_config.model_limit > 0 && results.size() >= m_config.model_limit) {
                        continue;
                    }

                    std::shared_ptr<Model> negative(*negative_it);
                    std::shared_ptr<Model> positive(*positive_it);
                    Model *model = new Model(feature, negative, positive, m_dataset);
                    model->identify(identifier);
                    model->translate_self(task.order());
                    translation_accessor negative_translation, positive_translation;
                    if ((**negative_it).identified() &&
                        m_graph.translations.find(negative_translation, std::make_pair(identifier, -(feature + 1)))) {
                        model->translate_negatives(negative_translation->second);
                    }
                    negative_translation.release();
                    if ((**positive_it).identified() &&
                        m_graph.translations.find(positive_translation, std::make_pair(identifier, feature + 1))) {
                        model->translate_positives(positive_translation->second);
                    }
                    positive_translation.release();
                    results.insert(model);
                }
            }
        }
    }
}