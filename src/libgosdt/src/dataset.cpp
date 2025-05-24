#include <cmath>
#include <configuration.hpp>
#include <dataset.hpp>
#include <ostream>
#include <stdexcept>
#include <utils/logging.hpp>

Dataset::Dataset(const Configuration &config, const Matrix<bool> &input_data, const Matrix<float> &cost_matrix,
                 const std::vector<std::set<size_t>> &feature_map)
    : m_config(config),
      m_number_rows(input_data.n_rows()),
      m_number_targets(cost_matrix.n_rows()),
      m_number_features(input_data.n_columns() - cost_matrix.n_rows()),
      m_feature_map(feature_map) {
    if (input_data.n_columns() == cost_matrix.n_rows() || input_data.n_columns() == 0) {
        throw std::invalid_argument(
            "During dataset processing, it was found that the provided dataset has no feature columns.");
    }

    if (cost_matrix.n_rows() != cost_matrix.n_columns() || cost_matrix.n_rows() == 0) {
        throw std::invalid_argument(
            "During dataset processing, it was found that the provided cost matrix was improperly formatted. The cost "
            "matrix must be a square matrix.");
    }

    if (input_data.n_rows() == 0) {
        throw std::invalid_argument("During dataset processing, it was found that the provided dataset has no rows.");
    }

    construct_bitmasks(input_data);
    construct_cost_matrices(cost_matrix);
    construct_majority_bitmask();
}

Dataset::Dataset(const Configuration &config, const Matrix<bool> &input_data, const Matrix<float> &cost_matrix,
                 const std::vector<std::set<size_t>> &feature_map, const Matrix<bool> &reference_matrix)
    : Dataset(config, input_data, cost_matrix, feature_map) {
    if (reference_matrix.n_columns() != m_number_targets || reference_matrix.n_rows() != m_number_rows) {
        throw std::invalid_argument(
            "During dataset processing, it was found that the provided reference matrix was improperly formatted. The "
            "reference matrix must have the same number of rows as the dataset and the number of columns must match "
            "that of the number of targets.");
    }

    construct_reference_bitmasks(reference_matrix);
}

Dataset::SummaryStatistics Dataset::summary_statistics(const Bitmask &capture_set, Bitmask &work_buffer) const {
    float support = static_cast<float>(capture_set.count()) / m_number_rows;
    // compute the distribution of each of the targets that are captured by the
    // given capture set.
    std::vector<size_t> distribution(m_number_targets, 0);
    for (size_t target = 0; target < m_number_targets; target++) {
        work_buffer = capture_set;
        m_col_view_targets[target].bit_and(work_buffer);
        distribution[target] = work_buffer.count();
    }

    // compute the loss incurred if the capture_set is left un-split and is
    // classified by it's majority target.
    float max_loss = std::numeric_limits<float>::max();
    size_t optimal_feature;
    for (size_t i = 0; i < m_number_targets; i++) {
        float cost = 0;
        for (size_t j = 0; j < m_number_targets; j++) {
            cost += m_cost_matrix(i, j) * distribution[j];
        }
        if (cost < max_loss) {
            max_loss = cost;
            optimal_feature = i;
        }
    }

    // compute the equivalent point loss for the capture_set
    float guaranteed_min_loss = 0;
    float max_cost_reduction = 0;
    float information = 0;
    for (size_t target = 0; target < m_number_targets; target++) {
        // Maximum cost difference accross different predictions
        max_cost_reduction += m_diff_costs[target] * distribution[target];

        // Cost of captured majority points with label target.
        work_buffer = capture_set;
        m_majority_bitmask.bit_and(work_buffer, false);
        m_col_view_targets[target].bit_and(work_buffer, false);
        guaranteed_min_loss += m_match_costs[target] * work_buffer.count();

        // cost of caputred minority points with label target.
        work_buffer = capture_set;
        m_majority_bitmask.bit_and(work_buffer, true);
        m_col_view_targets[target].bit_and(work_buffer, false);
        guaranteed_min_loss += m_mismatch_costs[target] * work_buffer.count();

        if (distribution[target] > 0) {
            float proportion = static_cast<float>(distribution[target]) / capture_set.count();
            information += -1 * proportion * log(proportion);
        }
    }

    // Because we are using floating point calculation, we hight have our
    // eqp_loss > max_loss in cases where they should be very close (or the
    // same). To avoid contradictions and maintain the invarian that eqp_los <=
    // max_loss, we correct for that here.
    guaranteed_min_loss = std::min(guaranteed_min_loss, max_loss);

    float best_min_loss = guaranteed_min_loss;
    if (m_reference_targets.has_value()) {
        best_min_loss = 0;
        for (size_t target = 0; target < m_number_targets; target++) {
            // cost of captured points with label target classified correctly by
            // reference model.
            work_buffer = capture_set;
            m_col_view_targets[target].bit_and(work_buffer, false);
            m_reference_targets.value()[target].bit_and(work_buffer, false);
            best_min_loss += m_match_costs[target] * work_buffer.count();

            // cost of captured points with label target mis-classified by the
            // reference model.
            work_buffer = capture_set;
            m_col_view_targets[target].bit_and(work_buffer, false);
            m_reference_targets.value()[target].bit_and(work_buffer, true);
            best_min_loss += m_mismatch_costs[target] * work_buffer.count();
        }
    }

    return {information, max_cost_reduction, max_loss, guaranteed_min_loss, best_min_loss, optimal_feature};
}

void Dataset::subset_inplace(Bitmask &capture_set, size_t feature_index, bool positive) const {
    m_col_view_features[feature_index].bit_and(capture_set, !positive);
}

float Dataset::distance(const Bitmask &capture_set, size_t i, size_t j, Bitmask &work_buffer) const {
    float positive_distance = 0, negative_distance = 0;
    for (size_t target = 0; target < m_number_targets; target++) {
        work_buffer = m_col_view_features[i];
        m_col_view_features[j].bit_xor(work_buffer, false);
        capture_set.bit_and(work_buffer, false);
        m_col_view_targets[target].bit_and(work_buffer, false);
        positive_distance += m_diff_costs[target] * work_buffer.count();

        work_buffer = m_col_view_features[i];
        m_col_view_features[j].bit_xor(work_buffer, true);
        capture_set.bit_and(work_buffer, false);
        m_col_view_targets[target].bit_and(work_buffer, false);
        negative_distance += m_diff_costs[target] * work_buffer.count();
    }
    return std::min(positive_distance, negative_distance);
}

size_t Dataset::original_feature(size_t binarized_feature_index) const {
    for (size_t i = 0; i < m_feature_map.size(); i++) {
        if (m_feature_map[i].find(binarized_feature_index) != m_feature_map[i].end()) {
            return i;
        }
    }
    gosdt_error("The binarized feature ", binarized_feature_index,
                " does not have an original feature index in the provided "
                "feature map.");
}

void Dataset::construct_bitmasks(const Matrix<bool> &input_data) {
    m_row_view_features.resize(m_number_rows, m_number_features);
    m_row_view_targets.resize(m_number_rows, m_number_targets);
    m_col_view_features.resize(m_number_features, m_number_rows);
    m_col_view_targets.resize(m_number_targets, m_number_rows);

    // constructing Bitmasks.
    for (size_t row = 0; row < m_number_rows; row++) {
        for (size_t col = 0; col < m_number_features; col++) {
            m_row_view_features[row].set(col, input_data(row, col));
            m_col_view_features[col].set(row, input_data(row, col));
        }

        for (size_t tar = 0; tar < m_number_targets; tar++) {
            m_row_view_targets[row].set(tar, input_data(row, m_number_features + tar));
            m_col_view_targets[tar].set(row, input_data(row, m_number_features + tar));
        }
    }
}

void Dataset::construct_cost_matrices(const Matrix<float> &cost_matrix) {
    m_cost_matrix = cost_matrix;
    auto max_costs = std::vector<float>(m_number_targets, -std::numeric_limits<float>::max());
    auto min_costs = std::vector<float>(m_number_targets, std::numeric_limits<float>::max());
    m_diff_costs.resize(m_number_targets, std::numeric_limits<float>::max());
    m_match_costs.resize(m_number_targets, 0.0f);
    m_mismatch_costs.resize(m_number_targets, std::numeric_limits<float>::max());

    for (size_t i = 0; i < m_number_targets; i++) {
        for (size_t j = 0; j < m_number_targets; j++) {
            max_costs[i] = std::max(max_costs[i], m_cost_matrix(j, i));
            min_costs[i] = std::min(min_costs[i], m_cost_matrix(j, i));
            if (i == j) {
                m_match_costs[i] = m_cost_matrix(j, i);
            } else {
                m_mismatch_costs[i] = std::min(m_mismatch_costs[i], m_cost_matrix(j, i));
            }
        }
        m_diff_costs[i] = max_costs[i] - min_costs[i];
    }
}

void Dataset::construct_majority_bitmask() {
    /*
     * `m_majority_bitmask` is a Bitmask of length `m_number_rows`. For row `i`
     * the value majority element `i` is True if the row maps to the MAJORITY
     * value for the given equivalence class of rows. This is because any
     * sample row in the dataset can appear multiple times with diferent target
     * values. The chosen MAJORITY value is selected through a cost
     * minimization calculation (this computation depends on our choice of cost
     * matrix).
     */

    // Frequency calculation
    std::unordered_map<Bitmask, std::vector<size_t>> target_distributions;
    for (size_t i = 0; i < m_number_rows; i++) {
        const Bitmask &id = m_row_view_features[i];
        if (target_distributions[id].size() != m_number_targets) {
            target_distributions[id].resize(m_number_targets, 0);
        }
        for (size_t j = 0; j < m_number_targets; j++) {
            target_distributions[id][j] += m_row_view_targets[i].get(j);
        }
    }

    // Cost minimization procedure
    std::unordered_map<Bitmask, size_t> cost_minimizers;
    for (const auto &[id, distribution] : target_distributions) {
        float min = std::numeric_limits<float>::max();
        size_t minimizer = 0;
        for (size_t i = 0; i < m_number_targets; i++) {
            float cost = 0;
            for (size_t j = 0; j < m_number_targets; j++) {
                cost += m_cost_matrix(i, j) * distribution[j];
            }

            if (cost < min) {
                min = cost;
                minimizer = i;
            }
        }
        cost_minimizers[id] = minimizer;
    }

    // Create majority bitmask
    m_majority_bitmask = Bitmask(m_number_rows, false);
    for (size_t i = 0; i < m_number_rows; i++) {
        const Bitmask &id = m_row_view_features[i];
        size_t minimizer = cost_minimizers[id];
        size_t empirical_target = m_row_view_targets[i].scan(0, true);
        if (empirical_target >= m_number_targets) {
            throw std::invalid_argument(
                "During dataset processing, a dataset row was found, which contains no target values.");
        }
        m_majority_bitmask.set(i, minimizer == empirical_target);
    }
}

void Dataset::construct_reference_bitmasks(const Matrix<bool> &reference_matrix) {
    m_reference_targets = std::vector<Bitmask>(m_number_rows, m_number_targets);
    for (size_t i = 0; i < m_number_rows; i++) {
        for (size_t j = 0; j < m_number_targets; j++) {
            m_reference_targets.value()[i].set(j, reference_matrix(i, j));
        }
    }
}

void Dataset::save(const std::string &filename) const {
    // Create Matrix<bool> for input data
    Matrix<bool> input_data(m_number_rows, m_number_features + m_number_targets);
    for (size_t i = 0; i < m_number_rows; i++) {
        for (size_t j = 0; j < m_number_features; j++) {
            input_data(i, j) = m_row_view_features[i].get(j);
        }
        for (size_t j = 0; j < m_number_targets; j++) {
            input_data(i, m_number_features + j) = m_row_view_targets[i].get(j);
        }
    }

    // Create Matrix<float> for cost matrix
    Matrix<float> cost_matrix(m_number_targets, m_number_targets);
    for (size_t i = 0; i < m_number_targets; i++) {
        for (size_t j = 0; j < m_number_targets; j++) {
            cost_matrix(i, j) = m_cost_matrix(i, j);
        }
    }

    // If there is a reference model, create Matrix<bool> for reference matrix
    Matrix<bool> reference_matrix;
    if (m_reference_targets.has_value()) {
        reference_matrix = Matrix<bool>(m_number_rows, m_number_targets);
        for (size_t i = 0; i < m_number_rows; i++) {
            for (size_t j = 0; j < m_number_targets; j++) {
                reference_matrix(i, j) = m_reference_targets.value()[i].get(j);
            }
        }
    }

    // Save the dataset
    std::ofstream file(filename);
    file << input_data << cost_matrix;
    if (m_reference_targets.has_value()) {
        file << true << std::endl;
        file << reference_matrix;
    } else {
        file << false << std::endl;
    }
    for (const auto &feature_set : m_feature_map) {
        for (const auto &feature : feature_set) {
            file << feature << " ";
        }
        file << std::endl;
    }

    file.close();
}

Dataset Dataset::load(const Configuration &config, const std::string &filename) {
    // Load the dataset
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("[Dataset] failed to open file for reading.");
    }

    Matrix<bool> input_data;
    Matrix<float> cost_matrix;
    Matrix<bool> reference_matrix;
    bool HAS_REFERENCE_MATRIX;

    file >> input_data >> cost_matrix >> HAS_REFERENCE_MATRIX;
    if (HAS_REFERENCE_MATRIX) {
        file >> reference_matrix;
    }
    // Feature map
    std::vector<std::set<size_t>> feature_map;
    std::string line;
    while (std::getline(file, line)) {
        std::set<size_t> feature_set;
        std::istringstream iss(line);
        size_t feature;
        while (iss >> feature) {
            feature_set.insert(feature);
        }
        feature_map.push_back(feature_set);
    }

    file.close();
    return HAS_REFERENCE_MATRIX ? Dataset(config, input_data, cost_matrix, feature_map, reference_matrix)
                                : Dataset(config, input_data, cost_matrix, feature_map);
}