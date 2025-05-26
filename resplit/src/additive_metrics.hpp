#ifndef ADDITIVE_METRICS_H
#define ADDITIVE_METRICS_H
#include <cstddef>
#include <functional>
#include <tuple>

struct Objective {
    Objective(const int falses, const int regularization);
    Objective() = default;

    int falses;
    int regularization;
    float objective;

    Objective operator+(const Objective &other) const {
        return Objective(falses + other.falses,
                         regularization + other.regularization);
    }

    std::tuple<float, int, int> to_tuple() const {
        return std::make_tuple(objective, falses, regularization);
    }

    bool operator==(const Objective &other) const {
        return objective == other.objective;
    }

    bool operator<(const Objective &other) const {
        return objective < other.objective;
    }

    bool operator<(const float &other) const { return objective < other; }
    bool operator<=(const float &other) const { return objective <= other; }
    bool operator>(const float &other) const { return objective > other; }
    bool operator>=(const float &other) const { return objective >= other; }
};

struct ValuesOfInterest {
    ValuesOfInterest() = default;
    ValuesOfInterest(const int TP, const int TN, const int regularization)
        : TP(TP), TN(TN), regularization(regularization){};

    int TP;
    int TN;
    int regularization;

    ValuesOfInterest operator+(const ValuesOfInterest &other) const {
        return ValuesOfInterest(TP + other.TP, TN + other.TN,
                                regularization + other.regularization);
    }

    bool operator==(const ValuesOfInterest &other) const {
        return TP == other.TP && TN == other.TN &&
               regularization == other.regularization;
    }

    size_t hash() const {
        size_t seed = 0;
        // boost::hash_combine(result, TP);
        // boost::hash_combine(result, TN);
        // boost::hash_combine(result, regularization);
        seed ^= TP + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        seed ^= TN + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        seed ^= regularization + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        return seed;
    }

    std::tuple<int, int, int> to_tuple() const {
        return std::make_tuple(TP, TN, regularization);
    }
};

struct ObjectiveHash {
    std::size_t operator()(const Objective &k) const {
        return std::hash<float>{}(k.objective);
    }
};

struct ObjectiveLess {
    bool operator()(const Objective &left, const Objective &right) const {
        return left.objective < right.objective;
    }
};
#endif