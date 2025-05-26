#include "task.hpp"

#include "graph.hpp"

Task::Task(void) {}

Task::Task(Bitmask const &capture_set, Bitmask const &feature_set, const Dataset &dataset, Bitmask &work_buffer) {
    this->_capture_set = capture_set;
    this->_feature_set = feature_set;
    this->_support = (float)(capture_set.count()) / (float)(dataset.m_number_rows);
    float const regularization = dataset.m_config.regularization;
    bool terminal = (this->_capture_set.count() <= 1) || (this->_feature_set.empty());

    // Careful, the following method modifies capture_set
    auto [info, potential, max_loss, min_loss, guaranteed_min_loss, optimal_feature] =
        dataset.summary_statistics(this->_capture_set, work_buffer);
    this->_information = info;

    // The Base objective is the loss incured if we don't split and choose the
    // optimal feature:
    this->_base_objective = max_loss + regularization;

    // Any tree with a better objective than the Base objective will have at
    // least two leaves:
    float lowerbound = std::min(this->_base_objective, min_loss + 2 * regularization);

    {
        this->_base_objective = max_loss + regularization;  // add 1*regularization because the max
                                                            // loss still uses one leaf
        // Since _base_objective corresponds to the best tree with just one
        // leaf, any tree with a better objective must use at least 2 leaves. So
        // we add 2*regularization to the min_loss in the calculation below
        float const lowerbound = std::min(this->_base_objective, min_loss + 2 * regularization);
        float const upperbound = this->_base_objective;

        // _guaranteed_lowerbound is a similar calculation to lowerbound, but
        // using guaranteed min loss
        this->_guaranteed_lowerbound = std::min(this->_base_objective, guaranteed_min_loss + 2 * regularization);

        // use lowerbound and upperbound to decide whether further splits are
        // possible
        if ((1.0 - min_loss < regularization)  // Insufficient maximum accuracy
            || (potential < 2 * regularization &&
                (1.0 - max_loss) < regularization))  // Leaf Support + Incremental Accuracy
        {                                            // Insufficient support and leaf accuracy
            // Node is provably not part of any optimal tree
            this->_lowerbound = this->_base_objective;
            this->_upperbound = this->_base_objective;
            this->_feature_set.clear();
        } else if (dataset.m_config.cart_lookahead_depth != 0 && capture_set.get_depth_budget() == 1) {
            // std::cout << "Using CART lookahead" << std::endl;
            // Set cart depth to max(1, depth_budget - cart_lookahead_depth), while being careful with unsigned chars
            unsigned char depth_for_cart = 1;
            if (dataset.m_config.cart_lookahead_depth < dataset.m_config.depth_budget) {
                depth_for_cart = dataset.m_config.depth_budget - dataset.m_config.cart_lookahead_depth + 1;
            }
            this -> _lowerbound = cart(this -> _capture_set, this -> _feature_set, dataset, work_buffer, depth_for_cart);
            this -> _upperbound = this -> _lowerbound;
            this -> _feature_set.clear();
        } else if (max_loss - min_loss < regularization  // Accuracy (also catches case where
                                                         // min_loss > max_loss, for
                                                         // Configuration::reference_LB)
                   || potential < 2 * regularization     // Leaf Support
                   || terminal ||
                   (dataset.m_config.depth_budget != 0 &&
                    capture_set.get_depth_budget() == 1)  // we are using depth constraints, and depth budget
                                                          // is exhausted
        ) {
            // Node is provably not an internal node of any optimal tree
            this->_lowerbound = this->_base_objective;
            this->_upperbound = this->_base_objective;
            this->_feature_set.clear();

        } else {
            // Node can be either an internal node or leaf
            this->_lowerbound = lowerbound;
            this->_upperbound = upperbound;
        }

        if (this->_lowerbound > this->_upperbound) {
            std::stringstream reason;
            reason << "Invalid Lowerbound (" << this->_lowerbound << ") or Upperbound (" << this->_upperbound << ")."
                   << std::endl;
            throw IntegrityViolation("Task::Task", reason.str());
        }
    }
}

float Task::support(void) const { return this->_support; }

float Task::information(void) const { return this->_information; }

float Task::base_objective(void) const { return this->_base_objective; }

float Task::uncertainty(void) const { return std::max((float)(0.0), upperbound() - lowerbound()); }

float Task::lowerbound(void) const { return this->_lowerbound; }
float Task::upperbound(void) const { return this->_upperbound; }
float Task::lowerscope(void) const { return this->_lowerscope; }
float Task::upperscope(void) const { return this->_upperscope; }
float Task::coverage() const { return this->_coverage; }
void Task::set_coverage(float new_coverage) { this->_coverage = new_coverage; }

double Task::guaranteed_lowerbound(const Configuration &config) {
    return (config.reference_LB) ? this->_guaranteed_lowerbound : this->_lowerbound;
}

Bitmask const &Task::capture_set(void) const { return this->_capture_set; }
Bitmask const &Task::feature_set(void) const { return this->_feature_set; }
//Bitmask &Task::identifier(void) { return this->_identifier; }
std::vector<int> &Task::order(void) { return this->_order; }

void Task::scope(float new_scope) {
    if (new_scope == 0) {
        return;
    }
    new_scope = std::max((float)(0.0), new_scope);
    this->_upperscope =
        this->_upperscope == std::numeric_limits<float>::max() ? new_scope : std::max(this->_upperscope, new_scope);
    this->_lowerscope =
        this->_lowerscope == -std::numeric_limits<float>::max() ? new_scope : std::min(this->_lowerscope, new_scope);
}

void Task::prune_feature(unsigned int index) { this->_feature_set.set(index, false); }

void Task::create_children(const Dataset &dataset, std::vector<Task> &neighbourhood, Bitmask &buffer, unsigned int id) {
    bool USING_DEPTH_BUDGET = this->_capture_set.get_depth_budget() != 0;
    bool conditions[2] = {false, true};
    Bitmask const &features = this->_feature_set;
    Bitmask work_buffer = Bitmask(this->capture_set().size(), false);
    for (int j_begin = 0, j_end = 0; features.scan_range(true, j_begin, j_end); j_begin = j_end) {
        for (int j = j_begin; j < j_end; ++j) {
            bool skip = false;
            for (unsigned int k = 0; k < 2; ++k) {
                buffer = this->_capture_set;
                dataset.subset_inplace(buffer, j, conditions[k]);
                if (USING_DEPTH_BUDGET) {
                    // In the case where we have a set depth_budget, children
                    // have one less than their parents.
                    buffer.set_depth_budget(buffer.get_depth_budget() - 1);
                }
                if (buffer.empty() || buffer == this->_capture_set) {
                    skip = true;
                    continue;
                }
                Task child(buffer, this->_feature_set, dataset, work_buffer);
                neighbourhood[2 * j + k] = child;
            }
            if (skip) {
                prune_feature(j);
            }
        }
    }
}

bool Task::update(const Configuration &config, float lower, float upper, int optimal_feature) {
    bool change = lower != this->_lowerbound || upper != this->_upperbound;
    this->_lowerbound = std::max(this->_lowerbound, lower);
    this->_upperbound = std::min(this->_upperbound, upper);
    this->_lowerbound = std::min(this->_upperbound, this->_lowerbound);

    this->_optimal_feature = optimal_feature;

    float regularization = config.regularization;
    if ((config.cancellation && 1.0 - this->_lowerbound < 0.0) ||
        this->_upperbound - this->_lowerbound <= std::numeric_limits<float>::epsilon()) {
        this->_lowerbound = this->_upperbound;
    }
    return change;
}

std::string Task::inspect(void) const {
    std::stringstream status;
    status << "Capture: " << this->_capture_set.to_string() << std::endl;
    status << "  Base: " << this->_base_objective << ", Bound: [" << this->_lowerbound << ", " << this->_upperbound
           << "]" << std::endl;
    status << "  Coverage: " << this->_coverage << ", Scope: [" << this->_lowerscope << ", " << this->_upperscope << "]"
           << std::endl;
    status << "  Feature: " << this->_feature_set.to_string() << std::endl;
    return status.str();
}

// expects depth to be at least 1. If depth is 1, the function will make no further splits
float Task::cart(Bitmask const &capture_set, Bitmask const &feature_set, Dataset const &dataset, Bitmask &work_buffer, unsigned char depth) {
    Bitmask left(dataset.m_number_rows);//dataset.m_number_features);
    Bitmask right(dataset.m_number_rows);//features);
    Dataset::SummaryStatistics stats = dataset.summary_statistics(capture_set, work_buffer);
    float base_risk = stats.max_loss + dataset.m_config.regularization;
    float base_info = stats.info;

    assert(stats.min_loss == stats.guaranteed_min_loss);

    if (depth == 1) {
        return base_risk;
    }

    int information_maximizer = -1;
    float information_gain = 0;
    for (int j_begin = 0, j_end = 0; feature_set.scan_range(true, j_begin, j_end); j_begin = j_end) {
        for (int j = j_begin; j < j_end; ++j) {
            left = capture_set;
            right = capture_set;
            dataset.subset_inplace(left, j, false);
            dataset.subset_inplace(right, j, true);

            if (left.empty() || right.empty()) {
                continue;
            }

            stats = dataset.summary_statistics(left, work_buffer);
            float left_info = stats.info;
            stats = dataset.summary_statistics(right, work_buffer);
            float right_info = stats.info;
            // Currently, calculating unscaled information gain (since that still preserves min/max)
            float gain = capture_set.count()*base_info - left.count()*left_info - right.count()*right_info;
            if (gain > information_gain) {
                information_maximizer = j;
                information_gain = gain;
            }
        }
    }

    if (information_maximizer == -1) {
        return base_risk;
    }

    left = capture_set;
    right = capture_set;
    dataset.subset_inplace(left, information_maximizer, false);
    dataset.subset_inplace(right, information_maximizer, true);
    float risk = cart(left, feature_set, dataset, work_buffer, depth - 1) + cart(right, feature_set, dataset, work_buffer, depth - 1);
    return std::min(risk, base_risk);
}