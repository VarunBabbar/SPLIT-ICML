#pragma once

#include <cstddef>
#include <string>

#include "nlohmann/json.hpp"
#include "dataset.hpp"

using json = nlohmann::json;

namespace gosdt {

enum class Status { CONVERGED, TIMEOUT, NON_CONVERGENCE, FALSE_CONVERGENCE, UNINITIALIZED };

struct Result {
    /// JSON Array of outputted models.
    std::string model = "";
    /// The number of problems in the graph.
    size_t graph_size = 0;
    /// The peak number of problems on the queue. todo(Ilias) Need to implement
    /// this.
    //  size_t peak_queue_size;
    /// The number of optimizer iterations.
    size_t n_iterations = 0;
    ///
    double lower_bound = 0.0;
    double upper_bound = 1.0;
    double model_loss = 0.0;
    double time_elapsed = 0.0;
    Status status = Status::UNINITIALIZED;
};

// todo(Ilias): Add documentation.
Result fit(const Dataset &dataset);
}  // namespace gosdt
