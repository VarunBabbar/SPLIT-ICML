#include <filesystem>
#include <iostream>

#include "gosdt.hpp"

namespace fs = std::filesystem;

// The GOSDT cli takes a folder path as input and reads the necessary files from 
// it and runs the GOSDT algorithm on the dataset.
int main(int argc, char *argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <debug_folder>" << std::endl;
        return 1;
    }

    // Open Debug folder and assert that the folder and necessary files exist:
    fs::path debug_folder(argv[1]);
    auto files = {"X.csv", "y.csv", "feature_names.csv",  "dataset.bin", "config.json"};
    

    if (!fs::exists(debug_folder) || !fs::is_directory(debug_folder)) {
        std::cerr << "Error: " << debug_folder << " is not a valid directory." << std::endl;
        return 1;
    }

    for (auto file : files) {
        if (!fs::exists(debug_folder / file)) {
            std::cerr << "Error: " << debug_folder / file << " does not exist." << std::endl;
            return 1;
        }
    }

    // Load the configuration and dataset file
    Configuration config = Configuration::load(debug_folder / "config.json");
    Dataset dataset = Dataset::load(config, debug_folder / "dataset.bin");

    gosdt::Result result = gosdt::fit(dataset);

    std::cout << "Model: " << result.model << std::endl;
    std::cout << "Graph Size: " << result.graph_size << std::endl;
    std::cout << "Number of Iterations: " << result.n_iterations << std::endl;
    std::cout << "Lower Bound: " << result.lower_bound << std::endl;
    std::cout << "Upper Bound: " << result.upper_bound << std::endl;
    std::cout << "Model Loss: " << result.model_loss << std::endl;
    std::cout << "Time Elapsed: " << result.time_elapsed << std::endl;
    std::cout << "Status: " << static_cast<int>(result.status) << std::endl;

    return 0;
}