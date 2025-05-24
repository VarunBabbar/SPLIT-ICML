#include <gosdt.hpp>
#include <optimizer.hpp>
#include <utils/logging.hpp>
#include <thread>
#include <atomic>

namespace gosdt {

Result fit(const Dataset &dataset) {
    // Initialize Optimization
    Result res;
    const Configuration &config = dataset.m_config;
    gosdt_verbose_log(config.verbose, "Using Configuration: ", config, "\nInitializing Optimization Framework.");

    // Initializes Local Buffers needed for efficient parallel computation
    // and emplace the first problem on the queue.
    Optimizer optimizer = Optimizer(config, dataset);
    std::atomic<Status> status = Status::CONVERGED;

    // Defines the work to be done by each thread
    /// @param wid:
    auto thread_work = [](size_t wid, Optimizer &optimizer, size_t &n_iterations, std::atomic<Status> *status) {
        try {
            while (optimizer.iterate(wid)) {
                n_iterations++;
            }
        } catch (IntegrityViolation &exception) {
            *status = Status::NON_CONVERGENCE;
            gosdt_log(exception.to_string());
            throw std::move(exception);
        }
    };

    // Perform Optimization
    if (config.verbose) {
        gosdt_log("Starting Optimization.");
    }
    optimizer.initialize();
    if (config.worker_limit > 1) {
        std::vector<std::thread> workers;
        std::vector<size_t> iterations(config.worker_limit);

        // Spin up the threads
        for (size_t i = 0; i < config.worker_limit; i++) {
            workers.emplace_back(thread_work, i, std::ref(optimizer), std::ref(iterations[i]), &status);
        }

        // Collect results from the threads
        for (size_t i = 0; i < config.worker_limit; i++) {
            workers[i].join();
            res.n_iterations += iterations[i];
        }
    } else {
        // In the single threaded case we can just call the work func instead.
        thread_work(0, optimizer, res.n_iterations, &status);
    }

    // Extract runtime statistics
    res.time_elapsed = optimizer.time_elapsed();
    res.graph_size = optimizer.size();
    auto [lb, ub] = optimizer.objective_boundary();
    res.lower_bound = lb;
    res.upper_bound = ub;
    res.status = Status::CONVERGED;
    gosdt_verbose_log(config.verbose, "Optimization Complete.\n", "Training Duration: ", res.time_elapsed, '\n',
                      "Number of Optimizer Iterations: ", res.n_iterations, '\n',
                      "Size of Problem Graph: ", res.graph_size, '\n', "Objective Boundary: [", res.lower_bound, ", ",
                      res.upper_bound, "]");
   
    // Check for timeout and non-convergence
    bool NOT_CONVERGED = res.lower_bound != res.upper_bound;
    if (NOT_CONVERGED) {
        // There might be a timeout.
        bool TIMEOUT = res.time_elapsed > static_cast<double>(config.time_limit);
        bool QUEUE_NONEMPTY = !optimizer.m_queue.empty();
        if (TIMEOUT || QUEUE_NONEMPTY) {
            gosdt_log("Possible timeout: ", res.time_elapsed, " Queue Size: ", optimizer.m_queue.size());
            res.status = Status::TIMEOUT;
        }
        // must have just been non-convergence.
        else {
            gosdt_log("Possible non-convergence: [", res.lower_bound, ", ", res.upper_bound, "]");
            res.status = Status::NON_CONVERGENCE;
        }

        if (config.diagnostics) {
            gosdt_log("Non-convergence detected. Beginning diagnosis.");
            optimizer.diagnose_non_convergence();
            gosdt_log("Diagnosis complete");
        }
    }

    // Extract models from the problem graph.
    std::unordered_set<Model> models;
    optimizer.models(models);

    // Check for False-convergence
    if (config.time_limit > 0 && models.empty()) {
        res.status = Status::FALSE_CONVERGENCE;
        if (config.diagnostics) {
            gosdt_log("False-convergence detected. Beginning diagnosis.");
            optimizer.diagnose_false_convergence();
            gosdt_log("Diagnosis complete");
        }
        return res;
    }

    gosdt_verbose_log(config.verbose, "Models Generated: ", models.size(), '\n', "Loss: ", models.begin()->loss(), '\n',
                      "Complexity: ", models.begin()->complexity());

    res.model_loss = models.begin()->loss();

    // Dump models to JSON
    json output = json::array();
    for (auto &model : models) {
        json object = json::object();
        model.to_json(object, dataset);
        output.push_back(object);
    }
    static const size_t SPACING = 2;
    res.model = output.dump(SPACING);

    return res;
}

}  // namespace gosdt
