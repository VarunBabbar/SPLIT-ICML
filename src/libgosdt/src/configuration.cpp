#include "configuration.hpp"

std::ostream &operator<<(std::ostream &ostream, const Configuration &config) {
    ostream << config.to_json().dump(4);
    return ostream;
}

Configuration Configuration::from_json(const json &object)
{
    Configuration config;
    config.regularization = object.at("regularization").get<float>();
    config.upperbound_guess = object.at("upperbound").get<float>();
    config.time_limit = object.at("time_limit").get<unsigned int>();
    config.worker_limit = object.at("worker_limit").get<unsigned int>();
    config.model_limit = object.at("model_limit").get<unsigned int>();
    config.verbose = object.at("verbose").get<bool>();
    config.diagnostics = object.at("diagnostics").get<bool>();
    config.depth_budget = object.at("depth_budget").get<unsigned char>();
    config.reference_LB = object.at("reference_LB").get<bool>();
    config.look_ahead = object.at("look_ahead").get<bool>();
    config.similar_support = object.at("similar_support").get<bool>();
    config.cancellation = object.at("cancellation").get<bool>();
    config.feature_transform = object.at("feature_transform").get<bool>();
    config.rule_list = object.at("rule_list").get<bool>();
    config.non_binary = object.at("non_binary").get<bool>();
    config.trace = object.at("trace").get<std::string>();
    config.tree = object.at("tree").get<std::string>();
    config.profile = object.at("profile").get<std::string>();
    return config;
}

nlohmann::json Configuration::to_json() const 
{
    nlohmann::json obj = nlohmann::json::object();
    obj["regularization"] = regularization;
    obj["upperbound"] = upperbound_guess;
    obj["time_limit"] = time_limit;
    obj["worker_limit"] = worker_limit;
    obj["model_limit"] = model_limit;
    obj["verbose"] = verbose;
    obj["diagnostics"] = diagnostics;
    obj["depth_budget"] = depth_budget;
    obj["reference_LB"] = reference_LB;
    obj["look_ahead"] = look_ahead;
    obj["similar_support"] = similar_support;
    obj["cancellation"] = cancellation;
    obj["feature_transform"] = feature_transform;
    obj["rule_list"] = rule_list;
    obj["non_binary"] = non_binary;
    obj["cart_lookahead_depth"] = cart_lookahead_depth;
    obj["trace"] = trace;
    obj["tree"] = tree;
    obj["profile"] = profile;
    return obj;
}

Configuration Configuration::load(const std::string &filename)
{
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + filename);
    }
    json object;
    file >> object;
    file.close();
    return Configuration::from_json(object);
}

void Configuration::save(const std::string &filename) const
{
    std::ofstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + filename);
    }
    file << to_json().dump(4);
    file.close();
}