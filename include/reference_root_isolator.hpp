#pragma once

#include "stationary_root_solver.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <vector>

namespace reference_roots {

enum class Status {
    resolved,
    unresolved,
    invalid_evaluation,
};

inline const char* status_name(Status status) {
    switch (status) {
        case Status::resolved:
            return "resolved";
        case Status::unresolved:
            return "unresolved";
        case Status::invalid_evaluation:
            return "invalid_evaluation";
    }
    return "unknown";
}

struct ReferenceConfig {
    std::vector<int> level_intervals = {512, 2048, 8192};
    int independent_scan_intervals = 16384;
    double pressure_tolerance = 1e-5;
    double derivative_tolerance = 1e-9;
    double density_tolerance = 1e-12;
    double density_agreement_tolerance = 1e-7;
    double root_merge_tolerance = 1e-9;
};

struct Result {
    Status status = Status::unresolved;
    std::string reason;
    std::vector<stationary_roots::Root> roots;
    std::vector<double> independent_roots;
    std::vector<stationary_roots::StationaryPoint> stationary_points;
    std::vector<std::vector<stationary_roots::Root>> level_roots;
    std::vector<int> level_root_counts;
    std::vector<int> level_stationary_counts;
    int pressure_evaluations = 0;
    int derivative_evaluations = 0;
    int independent_scan_evaluations = 0;
};

namespace detail {

inline bool same_density(double lhs, double rhs, double tolerance) {
    return std::abs(lhs - rhs) <= tolerance *
        std::max({1.0, std::abs(lhs), std::abs(rhs)});
}

inline double bisect_residual(
    const stationary_roots::Equation& equation,
    double lower,
    double upper,
    double value_lower,
    double pressure_tolerance,
    double density_tolerance,
    int& evaluations,
    bool& invalid) {
    double left = lower;
    double right = upper;
    double f_left = value_lower;
    for (int iteration = 0; iteration < 160; ++iteration) {
        const double middle = 0.5 * (left + right);
        const double f_middle = equation.pressure(middle) -
            equation.target_pressure;
        ++evaluations;
        if (!std::isfinite(f_middle)) {
            invalid = true;
            return std::numeric_limits<double>::quiet_NaN();
        }
        if (std::abs(f_middle) <= pressure_tolerance ||
            right - left <= density_tolerance * std::max(1.0, middle)) {
            return middle;
        }
        if ((f_left <= 0.0 && f_middle >= 0.0) ||
            (f_left >= 0.0 && f_middle <= 0.0)) {
            right = middle;
        } else {
            left = middle;
            f_left = f_middle;
        }
    }
    return 0.5 * (left + right);
}

inline std::vector<double> independent_residual_scan(
    const stationary_roots::Equation& equation,
    const ReferenceConfig& config,
    int& evaluations,
    bool& invalid) {
    std::vector<double> roots;
    const int intervals = std::max(1, config.independent_scan_intervals);
    double previous_density = equation.density_min;
    double previous_value = equation.pressure(previous_density) -
        equation.target_pressure;
    ++evaluations;
    if (!std::isfinite(previous_value)) {
        invalid = true;
        return roots;
    }
    if (std::abs(previous_value) <= config.pressure_tolerance) {
        roots.push_back(previous_density);
    }

    for (int index = 1; index <= intervals; ++index) {
        const double fraction = static_cast<double>(index) / intervals;
        const double density = equation.density_min + fraction *
            (equation.density_max - equation.density_min);
        const double value = equation.pressure(density) -
            equation.target_pressure;
        ++evaluations;
        if (!std::isfinite(value)) {
            invalid = true;
            return roots;
        }
        if ((previous_value < 0.0 && value > 0.0) ||
            (previous_value > 0.0 && value < 0.0)) {
            const double root = bisect_residual(
                equation, previous_density, density, previous_value,
                config.pressure_tolerance, config.density_tolerance,
                evaluations, invalid);
            if (std::isfinite(root)) {
                roots.push_back(root);
            }
        } else if (std::abs(value) <= config.pressure_tolerance) {
            roots.push_back(density);
        }
        previous_density = density;
        previous_value = value;
    }

    std::sort(roots.begin(), roots.end());
    roots.erase(
        std::unique(
            roots.begin(), roots.end(),
            [&](double lhs, double rhs) {
                return same_density(
                    lhs, rhs, config.root_merge_tolerance);
            }),
        roots.end());
    return roots;
}

inline bool root_sets_agree(
    const std::vector<stationary_roots::Root>& lhs,
    const std::vector<stationary_roots::Root>& rhs,
    double tolerance) {
    if (lhs.size() != rhs.size()) {
        return false;
    }
    for (std::size_t index = 0; index < lhs.size(); ++index) {
        if (!same_density(lhs[index].density, rhs[index].density, tolerance)) {
            return false;
        }
    }
    return true;
}

}  // namespace detail

inline Result isolate(
    const stationary_roots::Equation& equation,
    const ReferenceConfig& config = {}) {
    Result result;
    if (config.level_intervals.size() < 3 ||
        config.independent_scan_intervals < 1) {
        result.reason = "invalid_configuration";
        return result;
    }

    std::vector<stationary_roots::SolveResult> levels;
    levels.reserve(config.level_intervals.size());
    for (int intervals : config.level_intervals) {
        stationary_roots::SolverConfig solver_config;
        solver_config.linear_intervals = std::max(1, intervals);
        solver_config.log_intervals = std::max(0, intervals / 2);
        solver_config.pressure_tolerance = config.pressure_tolerance;
        solver_config.derivative_tolerance = config.derivative_tolerance;
        solver_config.density_tolerance = config.density_tolerance;
        solver_config.relative_merge_tolerance = config.root_merge_tolerance;
        auto level = stationary_roots::solve(equation, solver_config);
        result.pressure_evaluations += level.pressure_evaluations;
        result.derivative_evaluations += level.derivative_evaluations;
        result.level_root_counts.push_back(
            static_cast<int>(level.roots.size()));
        result.level_roots.push_back(level.roots);
        result.level_stationary_counts.push_back(
            static_cast<int>(level.stationary_points.size()));
        if (level.invalid_evaluation) {
            result.status = Status::invalid_evaluation;
            result.reason = "nonfinite_stationary_partition";
            return result;
        }
        levels.push_back(std::move(level));
    }

    result.roots = levels.back().roots;
    result.stationary_points = levels.back().stationary_points;
    const std::size_t first_converged_level = levels.size() - 1;
    for (std::size_t index = first_converged_level;
         index < levels.size(); ++index) {
        if (!detail::root_sets_agree(
                levels[index - 1].roots, levels[index].roots,
                config.density_agreement_tolerance)) {
            result.reason = "refinement_levels_disagree";
            return result;
        }
    }

    bool invalid_scan = false;
    const auto scan_roots = detail::independent_residual_scan(
        equation, config, result.independent_scan_evaluations, invalid_scan);
    result.independent_roots = scan_roots;
    if (invalid_scan) {
        result.status = Status::invalid_evaluation;
        result.reason = "nonfinite_independent_scan";
        return result;
    }
    for (double scan_root : scan_roots) {
        const bool matched = std::any_of(
            result.roots.begin(), result.roots.end(),
            [&](const stationary_roots::Root& root) {
                return detail::same_density(
                    scan_root, root.density,
                    10.0 * config.density_agreement_tolerance);
            });
        if (!matched) {
            result.reason = "independent_scan_found_additional_root";
            return result;
        }
    }

    for (const auto& root : result.roots) {
        const double residual = equation.pressure(root.density) -
            equation.target_pressure;
        ++result.pressure_evaluations;
        if (!std::isfinite(residual) ||
            std::abs(residual) > 2.0 * config.pressure_tolerance) {
            result.reason = "root_residual_exceeds_tolerance";
            return result;
        }
    }
    result.status = Status::resolved;
    result.reason = "converged_hierarchy";
    return result;
}

}  // namespace reference_roots
