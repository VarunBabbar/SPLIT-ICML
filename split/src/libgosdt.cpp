#include <fstream>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <matrix.hpp>
#include <configuration.hpp>
#include <gosdt.hpp>
#include <dataset.hpp>
#include <string>

// #define STRINGIFY(x) #x
// #define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;

PYBIND11_MODULE(_libgosdt, m) {

    using BoolMatrix = Matrix<bool>;
    using FloatMatrix = Matrix<float>;

    // Input binary matrix class
    py::class_<BoolMatrix>(m, "BoolMatrix", py::buffer_protocol())
        .def(py::init<size_t, size_t>())
        .def(py::init<size_t, size_t, bool>())
        .def("__getitem__",
                [](const BoolMatrix& bm, std::pair<size_t, size_t> tup) {
                    return bm(tup.first, tup.second);
                })
        .def("__setitem__",
                [](BoolMatrix& bm, std::pair<size_t, size_t> tup, bool value) {
                    bm(tup.first, tup.second) = value;
                })
        .def_buffer([](BoolMatrix &m) -> py::buffer_info {
                return py::buffer_info(
                        m.data(),
                        sizeof(bool),
                        py::format_descriptor<bool>::format(),
                        2,
                        { m.n_rows(), m.n_columns() },
                        { sizeof(bool) * m.n_columns(), sizeof(bool) }
                );
        });

    // float matrix class
    py::class_<FloatMatrix>(m, "FloatMatrix", py::buffer_protocol())
        .def(py::init<size_t, size_t>())
        .def(py::init<size_t, size_t, float>())
        .def("__getitem__",
                [](const FloatMatrix& bm, std::pair<size_t, size_t> tup) {
                    return bm(tup.first, tup.second);
                })
        .def("__setitem__",
                [](FloatMatrix& bm, std::pair<size_t, size_t> tup, bool value) {
                    bm(tup.first, tup.second) = value;
                })
        .def_buffer([](FloatMatrix &m) -> py::buffer_info {
                return py::buffer_info(
                        m.data(),
                        sizeof(float),
                        py::format_descriptor<float>::format(),
                        2,
                        { m.n_rows(), m.n_columns() },
                        { sizeof(float) * m.n_columns(), sizeof(float) }
                );
        });

    // Configuration class
    py::class_<Configuration>(m, "Configuration")
        .def(py::init<>())
        .def_readwrite("regularization",                &Configuration::regularization)
        .def_readwrite("upperbound",                    &Configuration::upperbound_guess)
        .def_readwrite("time_limit",                    &Configuration::time_limit)
        .def_readwrite("worker_limit",                  &Configuration::worker_limit)
        .def_readwrite("model_limit",                   &Configuration::model_limit)
        .def_readwrite("verbose",                       &Configuration::verbose)
        .def_readwrite("diagnostics",                   &Configuration::diagnostics)
        .def_readwrite("depth_budget",                  &Configuration::depth_budget)
        .def_readwrite("reference_LB",                  &Configuration::reference_LB)
        .def_readwrite("look_ahead",                    &Configuration::look_ahead)
        .def_readwrite("similar_support",               &Configuration::similar_support)
        .def_readwrite("cancellation",                  &Configuration::cancellation)
        .def_readwrite("feature_transform",             &Configuration::feature_transform)
        .def_readwrite("rule_list",                     &Configuration::rule_list)
        .def_readwrite("non_binary",                    &Configuration::non_binary)
        .def_readwrite("trace",                         &Configuration::trace)
        .def_readwrite("tree",                          &Configuration::tree)
        .def_readwrite("profile",                       &Configuration::profile)
        .def_readwrite("cart_lookahead_depth",          &Configuration::cart_lookahead_depth)
        .def("__repr__", [](const Configuration& config) { return config.to_json().dump(); })
        // Provides Pickling support for the Configuration class:
        .def(py::pickle(
            // __getstate__
            [](const Configuration& config) {
                // Return a tuple that fully encodes the state of the object
                return py::make_tuple(
                    config.regularization,
                    config.upperbound_guess,
                    config.time_limit,
                    config.worker_limit,
                    config.model_limit,
                    config.verbose,
                    config.diagnostics,
                    config.depth_budget,
                    config.reference_LB,
                    config.look_ahead,
                    config.similar_support,
                    config.cancellation,
                    config.feature_transform,
                    config.rule_list,
                    config.non_binary,
                    config.trace,
                    config.tree,
                    config.profile, 
                    config.cart_lookahead_depth
                );
            },
            // __setstate__
            [](const py::tuple& t) {
                if (t.size() != 18) {
                    throw std::runtime_error("Invalid state!");
                }
                Configuration config;
                config.regularization = t[0].cast<float>();
                config.upperbound_guess = t[1].cast<float>();
                config.time_limit = t[2].cast<unsigned int>();
                config.worker_limit = t[3].cast<unsigned int>();
                config.model_limit = t[4].cast<unsigned int>();
                config.verbose = t[5].cast<bool>();
                config.diagnostics = t[6].cast<bool>();
                config.depth_budget = t[7].cast<unsigned char>();
                config.reference_LB = t[8].cast<bool>();
                config.look_ahead = t[9].cast<bool>();
                config.similar_support = t[10].cast<bool>();
                config.cancellation = t[11].cast<bool>();
                config.feature_transform = t[12].cast<bool>();
                config.rule_list = t[13].cast<bool>();
                config.non_binary = t[14].cast<bool>();
                config.trace = t[15].cast<std::string>();
                config.tree = t[16].cast<std::string>();
                config.profile = t[17].cast<std::string>();
                config.cart_lookahead_depth = t[18].cast<unsigned char>();
                return config;
            }
        ))
        .def("save", &Configuration::save);

    // gosdt::Result Class
    py::class_<gosdt::Result>(m, "GOSDTResult")
        .def(py::init<gosdt::Result>())
        .def_readonly("model",          &gosdt::Result::model)
        .def_readonly("graph_size",     &gosdt::Result::graph_size)
        .def_readonly("n_iterations",   &gosdt::Result::n_iterations)
        .def_readonly("lowerbound",     &gosdt::Result::lower_bound)
        .def_readonly("upperbound",     &gosdt::Result::upper_bound)
        .def_readonly("model_loss",     &gosdt::Result::model_loss)
        .def_readonly("time",           &gosdt::Result::time_elapsed)
        .def_readonly("status",         &gosdt::Result::status)
        .def(py::pickle(
            [](const gosdt::Result& result) {
                return py::make_tuple(
                    result.model,
                    result.graph_size,
                    result.n_iterations,
                    result.lower_bound,
                    result.upper_bound,
                    result.model_loss,
                    result.time_elapsed,
                    result.status
                );
            },
            [](const py::tuple& t) {
                if (t.size() != 8) {
                    throw std::runtime_error("Invalid state!");
                }
                gosdt::Result result;
                result.model = t[0].cast<std::string>();
                result.graph_size = t[1].cast<size_t>();
                result.n_iterations = t[2].cast<size_t>();
                result.lower_bound = t[3].cast<double>();
                result.upper_bound = t[4].cast<double>();
                result.model_loss = t[5].cast<double>();
                result.time_elapsed = t[6].cast<double>();
                result.status = t[7].cast<gosdt::Status>();
                return result;
            }
        ));

    // gosdt::fit function
    m.def("gosdt_fit", &gosdt::fit);

    // Define Status enum
    py::enum_<gosdt::Status>(m, "Status")
        .value("CONVERGED",         gosdt::Status::CONVERGED)
        .value("TIMEOUT",           gosdt::Status::TIMEOUT)
        .value("NON_CONVERGENCE",   gosdt::Status::NON_CONVERGENCE)
        .value("FALSE_CONVERGENCE", gosdt::Status::FALSE_CONVERGENCE)
        .value("UNINITIALIZED",     gosdt::Status::UNINITIALIZED)
        .export_values();

    // Encoding class for translating between original features and binarized features.
    py::class_<Dataset>(m, "Dataset")
        .def(py::init<const Configuration&, const Matrix<bool>&, const Matrix<float>&, const std::vector<std::set<size_t>>&>())
        .def(py::init<const Configuration&, const Matrix<bool>&, const Matrix<float>&, const std::vector<std::set<size_t>>&, const Matrix<bool>&>())
        .def_readonly("n_rows",     &Dataset::m_number_rows)
        .def_readonly("n_features", &Dataset::m_number_features)
        .def_readonly("n_targets",  &Dataset::m_number_targets)
        .def("save",                &Dataset::save);
        // .def_static("load",         &Dataset::load);

}
