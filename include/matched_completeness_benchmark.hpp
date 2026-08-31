#pragma once

#include "full_solver_benchmark.hpp"
#include "homotopy_solver.hpp"

#include <algorithm>
#include <cmath>
#include <sstream>
#include <string>
#include <vector>

namespace matched_completeness {

struct EvaluationCounter {
    long long pressure_calls = 0;
    long long derivative_calls = 0;
    long long lambda_derivative_calls = 0;

    long long total() const {
        return pressure_calls + derivative_calls +
            lambda_derivative_calls;
    }
};

inline homotopy::EoSHomotopyInterface counted_eos(
    const homotopy::EoSHomotopyInterface& source,
    EvaluationCounter& counter) {
    auto counted = source;
    counted.P = [function = source.P, &counter](double rho, double lambda) {
        ++counter.pressure_calls;
        return function(rho, lambda);
    };
    counted.dPdrho =
        [function = source.dPdrho, &counter](double rho, double lambda) {
            ++counter.derivative_calls;
            return function(rho, lambda);
        };
    counted.dPdlam =
        [function = source.dPdlam, &counter](double rho, double lambda) {
            ++counter.lambda_derivative_calls;
            return function(rho, lambda);
        };
    if (source.evaluate_all) {
        counted.evaluate_all =
            [function = source.evaluate_all, &counter](
                double rho, double lambda) {
                ++counter.pressure_calls;
                ++counter.derivative_calls;
                ++counter.lambda_derivative_calls;
                return function(rho, lambda);
            };
    }
    return counted;
}

struct ParsedReferenceRoot {
    bool valid = false;
    std::string state_id;
    solver_benchmark::Root root;
};

inline ParsedReferenceRoot parse_reference_root_row(
    const std::string& line) {
    std::vector<std::string> fields;
    std::stringstream input(line);
    std::string field;
    while (std::getline(input, field, ',')) {
        fields.push_back(field);
    }
    if (fields.size() != 8 || fields[1] != "reference") {
        return {};
    }
    try {
        ParsedReferenceRoot parsed;
        parsed.valid = true;
        parsed.state_id = fields[0];
        parsed.root.density = std::stod(fields[3]);
        parsed.root.residual = std::stod(fields[4]);
        parsed.root.derivative = std::stod(fields[5]);
        parsed.root.mechanically_stable = std::stoi(fields[6]) != 0;
        parsed.root.tangent = std::stoi(fields[7]) != 0;
        return parsed;
    } catch (...) {
        return {};
    }
}

inline double quantile(std::vector<double> values, double probability) {
    if (values.empty() || !std::isfinite(probability)) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    std::sort(values.begin(), values.end());
    const double position = std::clamp(probability, 0.0, 1.0) *
        static_cast<double>(values.size() - 1);
    const std::size_t lower = static_cast<std::size_t>(std::floor(position));
    const std::size_t upper = static_cast<std::size_t>(std::ceil(position));
    const double fraction = position - static_cast<double>(lower);
    return values[lower] + fraction * (values[upper] - values[lower]);
}

}  // namespace matched_completeness
