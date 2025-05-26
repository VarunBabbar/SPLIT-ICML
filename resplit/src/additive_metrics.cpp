#include "additive_metrics.hpp"
#include "state.hpp"
#include "configuration.hpp"

Objective::Objective(const int falses, const int regularization)
    : falses(falses), regularization(regularization),
      objective(falses * State::dataset.get_mismatch_cost() +
                regularization * Configuration::regularization){};