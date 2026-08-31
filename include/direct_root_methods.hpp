#pragma once

#include "stationary_root_solver.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <vector>

namespace direct_roots {

struct RootRecord {
    double density = std::numeric_limits<double>::quiet_NaN();
    double residual = std::numeric_limits<double>::quiet_NaN();
    double derivative = std::numeric_limits<double>::quiet_NaN();
    bool mechanically_stable = false;
};

struct MethodResult {
    bool success = false;
    std::string failure_reason;
    std::vector<RootRecord> roots;
    int pressure_evaluations = 0;
    int derivative_evaluations = 0;
};

struct RootRefinementResult {
    bool success = false;
    RootRecord root;
    int pressure_evaluations = 0;
    int derivative_evaluations = 0;
};

struct MethodConfig {
    int scan_intervals = 4096;
    int stationary_intervals = 2048;
    int maximum_newton_iterations = 80;
    int maximum_line_search_iterations = 20;
    double pressure_tolerance = 1e-6;
    double derivative_tolerance = 1e-10;
    double density_tolerance = 1e-11;
    double relative_merge_tolerance = 1e-8;
};

inline RootRefinementResult refine_root(
    const stationary_roots::Equation& equation,
    double initial_density,
    const MethodConfig& config = {}) {
    RootRefinementResult result;
    const double lower = equation.density_min;
    const double upper = equation.density_max;
    const double margin = 16.0 * std::numeric_limits<double>::epsilon() *
        std::max(1.0, upper);
    if (!std::isfinite(initial_density) || lower >= upper) {
        return result;
    }
    double density = std::clamp(
        initial_density, lower + margin, upper - margin);

    for (int iteration = 0;
         iteration < config.maximum_newton_iterations;
         ++iteration) {
        const double residual =
            equation.pressure(density) - equation.target_pressure;
        const double derivative = equation.derivative(density);
        ++result.pressure_evaluations;
        ++result.derivative_evaluations;
        if (!std::isfinite(residual) || !std::isfinite(derivative)) {
            return result;
        }
        if (std::abs(residual) <= config.pressure_tolerance) {
            result.success = true;
            result.root = {
                density, residual, derivative, derivative > 0.0};
            return result;
        }
        if (std::abs(derivative) <= config.derivative_tolerance) {
            return result;
        }

        double candidate = std::clamp(
            density - residual / derivative,
            lower + margin,
            upper - margin);
        double candidate_residual =
            equation.pressure(candidate) - equation.target_pressure;
        ++result.pressure_evaluations;
        for (int line_search = 0;
             line_search < config.maximum_line_search_iterations &&
             (!std::isfinite(candidate_residual) ||
              std::abs(candidate_residual) >= std::abs(residual));
             ++line_search) {
            candidate = 0.5 * (density + candidate);
            candidate_residual =
                equation.pressure(candidate) - equation.target_pressure;
            ++result.pressure_evaluations;
        }
        if (!std::isfinite(candidate_residual) ||
            std::abs(candidate_residual) >= std::abs(residual)) {
            return result;
        }
        density = candidate;
    }
    return result;
}

namespace detail {

inline bool same_density(
    double lhs,
    double rhs,
    const MethodConfig& config) {
    return std::abs(lhs - rhs) <= config.relative_merge_tolerance *
        std::max({1.0, std::abs(lhs), std::abs(rhs)});
}

inline void add_root(
    MethodResult& result,
    double density,
    double residual,
    double derivative,
    const MethodConfig& config) {
    if (!std::isfinite(density) || !std::isfinite(residual) ||
        !std::isfinite(derivative) ||
        std::abs(residual) > 2.0 * config.pressure_tolerance) {
        return;
    }
    for (auto& root : result.roots) {
        if (same_density(root.density, density, config)) {
            if (std::abs(residual) < std::abs(root.residual)) {
                root = {density, residual, derivative, derivative > 0.0};
            }
            return;
        }
    }
    result.roots.push_back({density, residual, derivative, derivative > 0.0});
}

inline void finish(MethodResult& result) {
    std::sort(
        result.roots.begin(), result.roots.end(),
        [](const RootRecord& lhs, const RootRecord& rhs) {
            return lhs.density < rhs.density;
        });
    result.success = !result.roots.empty();
    if (!result.success && result.failure_reason.empty()) {
        result.failure_reason = "no_converged_root";
    }
}

inline double bisect(
    const stationary_roots::Equation& equation,
    double lower,
    double upper,
    double value_lower,
    const MethodConfig& config,
    MethodResult& result) {
    double left = lower;
    double right = upper;
    double f_left = value_lower;
    for (int iteration = 0; iteration < 140; ++iteration) {
        const double middle = 0.5 * (left + right);
        const double f_middle =
            equation.pressure(middle) - equation.target_pressure;
        ++result.pressure_evaluations;
        if (!std::isfinite(f_middle)) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        if (std::abs(f_middle) <= config.pressure_tolerance ||
            right - left <= config.density_tolerance *
                std::max(1.0, middle)) {
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

}  // namespace detail

inline MethodResult uniform_scan(
    const stationary_roots::Equation& equation,
    const MethodConfig& config = {}) {
    MethodResult result;
    const int intervals = std::max(1, config.scan_intervals);
    double previous_density = equation.density_min;
    double previous_value =
        equation.pressure(previous_density) - equation.target_pressure;
    ++result.pressure_evaluations;
    if (!std::isfinite(previous_value)) {
        result.failure_reason = "nonfinite_evaluation";
        return result;
    }

    for (int index = 1; index <= intervals; ++index) {
        const double fraction = static_cast<double>(index) / intervals;
        const double density = equation.density_min + fraction *
            (equation.density_max - equation.density_min);
        const double value = equation.pressure(density) -
            equation.target_pressure;
        ++result.pressure_evaluations;
        if (!std::isfinite(value)) {
            result.failure_reason = "nonfinite_evaluation";
            return result;
        }
        if ((previous_value < 0.0 && value > 0.0) ||
            (previous_value > 0.0 && value < 0.0) ||
            std::abs(value) <= config.pressure_tolerance) {
            const double root = std::abs(value) <= config.pressure_tolerance
                ? density
                : detail::bisect(
                      equation, previous_density, density, previous_value,
                      config, result);
            if (std::isfinite(root)) {
                const double residual =
                    equation.pressure(root) - equation.target_pressure;
                const double derivative = equation.derivative(root);
                ++result.pressure_evaluations;
                ++result.derivative_evaluations;
                detail::add_root(
                    result, root, residual, derivative, config);
            }
        }
        previous_density = density;
        previous_value = value;
    }
    detail::finish(result);
    return result;
}

inline MethodResult stationary_partition(
    const stationary_roots::Equation& equation,
    const MethodConfig& config = {}) {
    MethodResult result;
    stationary_roots::SolverConfig solver_config;
    solver_config.linear_intervals = std::max(1, config.stationary_intervals);
    solver_config.log_intervals =
        std::max(0, config.stationary_intervals / 2);
    solver_config.pressure_tolerance = config.pressure_tolerance;
    solver_config.derivative_tolerance = config.derivative_tolerance;
    solver_config.density_tolerance = config.density_tolerance;
    solver_config.relative_merge_tolerance = config.relative_merge_tolerance;
    const auto solved = stationary_roots::solve(equation, solver_config);
    result.pressure_evaluations = solved.pressure_evaluations;
    result.derivative_evaluations = solved.derivative_evaluations;
    if (solved.invalid_evaluation) {
        result.failure_reason = "nonfinite_evaluation";
        return result;
    }
    for (const auto& root : solved.roots) {
        detail::add_root(
            result, root.density, root.residual, root.derivative, config);
    }
    detail::finish(result);
    return result;
}

inline MethodResult newton_six_start(
    const stationary_roots::Equation& equation,
    const MethodConfig& config = {},
    bool safeguarded = false) {
    MethodResult result;
    constexpr double fractions[] = {0.0, 0.08, 0.28, 0.50, 0.72, 1.0};
    const double lower = equation.density_min;
    const double upper = equation.density_max;
    const double margin = 16.0 * std::numeric_limits<double>::epsilon() *
        std::max(1.0, upper);

    for (double fraction : fractions) {
        double density = lower + fraction * (upper - lower);
        density = std::clamp(density, lower + margin, upper - margin);
        for (int iteration = 0;
             iteration < config.maximum_newton_iterations;
             ++iteration) {
            const double residual =
                equation.pressure(density) - equation.target_pressure;
            const double derivative = equation.derivative(density);
            ++result.pressure_evaluations;
            ++result.derivative_evaluations;
            if (!std::isfinite(residual) || !std::isfinite(derivative)) {
                break;
            }
            if (std::abs(residual) <= config.pressure_tolerance) {
                detail::add_root(
                    result, density, residual, derivative, config);
                break;
            }
            if (std::abs(derivative) <= config.derivative_tolerance) {
                break;
            }

            double candidate = density - residual / derivative;
            if (!safeguarded) {
                if (!std::isfinite(candidate) || candidate <= lower ||
                    candidate >= upper) {
                    break;
                }
            } else {
                candidate = std::clamp(candidate, lower + margin, upper - margin);
                double candidate_residual = equation.pressure(candidate) -
                    equation.target_pressure;
                ++result.pressure_evaluations;
                for (int line_search = 0;
                     line_search < config.maximum_line_search_iterations &&
                     (!std::isfinite(candidate_residual) ||
                      std::abs(candidate_residual) >= std::abs(residual));
                     ++line_search) {
                    candidate = 0.5 * (density + candidate);
                    candidate_residual = equation.pressure(candidate) -
                        equation.target_pressure;
                    ++result.pressure_evaluations;
                }
                if (!std::isfinite(candidate_residual) ||
                    std::abs(candidate_residual) >= std::abs(residual)) {
                    break;
                }
            }
            if (std::abs(candidate - density) <= config.density_tolerance *
                    std::max(1.0, density)) {
                density = candidate;
                const double final_residual =
                    equation.pressure(density) - equation.target_pressure;
                const double final_derivative = equation.derivative(density);
                ++result.pressure_evaluations;
                ++result.derivative_evaluations;
                detail::add_root(
                    result, density, final_residual, final_derivative, config);
                break;
            }
            density = candidate;
        }
    }
    detail::finish(result);
    return result;
}

}  // namespace direct_roots
