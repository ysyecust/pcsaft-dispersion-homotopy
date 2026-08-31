#pragma once

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <utility>
#include <vector>

namespace stationary_roots {

struct Equation {
    std::function<double(double density)> pressure;
    std::function<double(double density)> derivative;
    double target_pressure = 0.0;
    double density_min = 0.0;
    double density_max = 0.0;
    double derivative_scale = 1.0;
};

struct SolverConfig {
    int linear_intervals = 2048;
    int log_intervals = 1024;
    int max_refinement_iterations = 120;
    int local_minimum_iterations = 100;
    double pressure_tolerance = 1e-6;
    double derivative_tolerance = 1e-8;
    double density_tolerance = 1e-11;
    double relative_merge_tolerance = 1e-9;
};

struct StationaryPoint {
    double density = std::numeric_limits<double>::quiet_NaN();
    double derivative = std::numeric_limits<double>::quiet_NaN();
    double pressure_residual = std::numeric_limits<double>::quiet_NaN();
    double bracket_lower = std::numeric_limits<double>::quiet_NaN();
    double bracket_upper = std::numeric_limits<double>::quiet_NaN();
    bool isolated_by_sign_change = false;
};

struct Root {
    double density = std::numeric_limits<double>::quiet_NaN();
    double residual = std::numeric_limits<double>::quiet_NaN();
    double derivative = std::numeric_limits<double>::quiet_NaN();
    double bracket_lower = std::numeric_limits<double>::quiet_NaN();
    double bracket_upper = std::numeric_limits<double>::quiet_NaN();
    bool mechanically_stable = false;
    bool at_stationary_point = false;
};

struct SolveResult {
    bool success = false;
    bool invalid_evaluation = false;
    int pressure_evaluations = 0;
    int derivative_evaluations = 0;
    std::vector<StationaryPoint> stationary_points;
    std::vector<Root> roots;
};

namespace detail {

inline double scaled_density_tolerance(
    double density,
    const SolverConfig& config) {
    return config.density_tolerance * std::max(1.0, std::abs(density));
}

inline bool same_density(
    double lhs,
    double rhs,
    const SolverConfig& config) {
    return std::abs(lhs - rhs) <= config.relative_merge_tolerance *
        std::max({1.0, std::abs(lhs), std::abs(rhs)});
}

inline std::vector<double> build_density_mesh(
    const Equation& equation,
    const SolverConfig& config) {
    std::vector<double> mesh;
    mesh.reserve(static_cast<std::size_t>(
        std::max(1, config.linear_intervals) +
        std::max(0, config.log_intervals) + 2));

    const int linear_intervals = std::max(1, config.linear_intervals);
    for (int index = 0; index <= linear_intervals; ++index) {
        const double fraction =
            static_cast<double>(index) / linear_intervals;
        mesh.push_back(
            equation.density_min +
            fraction * (equation.density_max - equation.density_min));
    }

    if (config.log_intervals > 0 && equation.density_max > 0.0) {
        const double positive_lower = std::max(
            equation.density_min,
            std::max(std::numeric_limits<double>::min(),
                     equation.density_max * 1e-12));
        if (positive_lower < equation.density_max) {
            const double log_lower = std::log(positive_lower);
            const double log_upper = std::log(equation.density_max);
            for (int index = 0; index <= config.log_intervals; ++index) {
                const double fraction =
                    static_cast<double>(index) / config.log_intervals;
                mesh.push_back(
                    std::exp(log_lower + fraction * (log_upper - log_lower)));
            }
        }
    }

    std::sort(mesh.begin(), mesh.end());
    mesh.erase(
        std::unique(
            mesh.begin(), mesh.end(),
            [](double lhs, double rhs) {
                return std::abs(lhs - rhs) <=
                    8.0 * std::numeric_limits<double>::epsilon() *
                    std::max({1.0, std::abs(lhs), std::abs(rhs)});
            }),
        mesh.end());
    return mesh;
}

template <typename Function>
inline double bisect_sign_change(
    Function&& function,
    double lower,
    double upper,
    double value_lower,
    double value_upper,
    double value_tolerance,
    const SolverConfig& config) {
    if (std::abs(value_lower) <= value_tolerance) {
        return lower;
    }
    if (std::abs(value_upper) <= value_tolerance) {
        return upper;
    }

    double left = lower;
    double right = upper;
    double f_left = value_lower;
    for (int iteration = 0;
         iteration < config.max_refinement_iterations;
         ++iteration) {
        const double middle = 0.5 * (left + right);
        const double f_middle = function(middle);
        if (!std::isfinite(f_middle)) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        if (std::abs(f_middle) <= value_tolerance ||
            std::abs(right - left) <=
                scaled_density_tolerance(middle, config)) {
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

template <typename Function>
inline double minimize_absolute_value(
    Function&& function,
    double lower,
    double upper,
    const SolverConfig& config) {
    constexpr double inverse_phi = 0.6180339887498948482;
    double left = lower;
    double right = upper;
    double x1 = right - inverse_phi * (right - left);
    double x2 = left + inverse_phi * (right - left);
    double f1 = std::abs(function(x1));
    double f2 = std::abs(function(x2));

    for (int iteration = 0;
         iteration < config.local_minimum_iterations;
         ++iteration) {
        if (!std::isfinite(f1) || !std::isfinite(f2)) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        if (std::abs(right - left) <=
            scaled_density_tolerance(0.5 * (left + right), config)) {
            break;
        }
        if (f1 <= f2) {
            right = x2;
            x2 = x1;
            f2 = f1;
            x1 = right - inverse_phi * (right - left);
            f1 = std::abs(function(x1));
        } else {
            left = x1;
            x1 = x2;
            f1 = f2;
            x2 = left + inverse_phi * (right - left);
            f2 = std::abs(function(x2));
        }
    }
    return f1 <= f2 ? x1 : x2;
}

}  // namespace detail

inline SolveResult solve(
    const Equation& equation,
    const SolverConfig& config = {}) {
    SolveResult result;
    if (!equation.pressure || !equation.derivative ||
        !std::isfinite(equation.density_min) ||
        !std::isfinite(equation.density_max) ||
        !(equation.density_max > equation.density_min) ||
        config.linear_intervals < 1 || config.log_intervals < 0) {
        return result;
    }

    const double derivative_tolerance = config.derivative_tolerance *
        std::max(1.0, std::abs(equation.derivative_scale));
    const double derivative_refinement_tolerance = std::max(
        1e-12,
        1e-14 * std::max(1.0, std::abs(equation.derivative_scale)));
    SolverConfig stationary_refinement_config = config;
    stationary_refinement_config.density_tolerance = std::min(
        config.density_tolerance,
        1e-14);

    auto pressure_residual = [&](double density) {
        ++result.pressure_evaluations;
        const double value = equation.pressure(density) -
            equation.target_pressure;
        if (!std::isfinite(value)) {
            result.invalid_evaluation = true;
        }
        return value;
    };
    auto derivative = [&](double density) {
        ++result.derivative_evaluations;
        const double value = equation.derivative(density);
        if (!std::isfinite(value)) {
            result.invalid_evaluation = true;
        }
        return value;
    };

    const auto mesh = detail::build_density_mesh(equation, config);
    if (mesh.size() < 2) {
        return result;
    }

    std::vector<double> derivative_values(mesh.size());
    for (std::size_t index = 0; index < mesh.size(); ++index) {
        derivative_values[index] = derivative(mesh[index]);
    }
    if (result.invalid_evaluation) {
        return result;
    }

    auto add_stationary = [&](double density,
                              double lower,
                              double upper,
                              bool sign_change) {
        if (!std::isfinite(density) ||
            density < equation.density_min ||
            density > equation.density_max) {
            return;
        }
        for (auto& existing : result.stationary_points) {
            const bool local_search_rediscovered_sign_change =
                existing.isolated_by_sign_change && !sign_change &&
                density >= existing.bracket_lower &&
                density <= existing.bracket_upper;
            if (detail::same_density(existing.density, density, config) ||
                local_search_rediscovered_sign_change) {
                existing.bracket_lower = std::min(existing.bracket_lower, lower);
                existing.bracket_upper = std::max(existing.bracket_upper, upper);
                existing.isolated_by_sign_change =
                    existing.isolated_by_sign_change || sign_change;
                return;
            }
        }
        const double d_value = derivative(density);
        if (!std::isfinite(d_value) ||
            std::abs(d_value) > derivative_tolerance) {
            return;
        }
        const double residual = pressure_residual(density);
        result.stationary_points.push_back({
            density,
            d_value,
            residual,
            lower,
            upper,
            sign_change,
        });
    };

    for (std::size_t index = 1; index < mesh.size(); ++index) {
        const double left_value = derivative_values[index - 1];
        const double right_value = derivative_values[index];
        if ((left_value <= 0.0 && right_value >= 0.0) ||
            (left_value >= 0.0 && right_value <= 0.0)) {
            const double density = detail::bisect_sign_change(
                derivative,
                mesh[index - 1],
                mesh[index],
                left_value,
                right_value,
                derivative_refinement_tolerance,
                stationary_refinement_config);
            add_stationary(
                density, mesh[index - 1], mesh[index], true);
        }
    }

    for (std::size_t index = 1; index + 1 < mesh.size(); ++index) {
        const double previous = std::abs(derivative_values[index - 1]);
        const double current = std::abs(derivative_values[index]);
        const double next = std::abs(derivative_values[index + 1]);
        if (current <= previous && current <= next) {
            const double density = detail::minimize_absolute_value(
                derivative, mesh[index - 1], mesh[index + 1], config);
            if (std::isfinite(density)) {
                const double d_value = derivative(density);
                if (std::isfinite(d_value) &&
                    std::abs(d_value) <= derivative_tolerance) {
                    add_stationary(
                        density, mesh[index - 1], mesh[index + 1], false);
                }
            }
        }
    }

    std::sort(
        result.stationary_points.begin(),
        result.stationary_points.end(),
        [](const StationaryPoint& lhs, const StationaryPoint& rhs) {
            return lhs.density < rhs.density;
        });

    std::vector<double> boundaries;
    boundaries.reserve(result.stationary_points.size() + 2);
    boundaries.push_back(equation.density_min);
    for (const auto& stationary : result.stationary_points) {
        boundaries.push_back(stationary.density);
    }
    boundaries.push_back(equation.density_max);

    auto is_stationary_density = [&](double density) {
        return std::any_of(
            result.stationary_points.begin(),
            result.stationary_points.end(),
            [&](const StationaryPoint& stationary) {
                return detail::same_density(
                    stationary.density, density, config);
            });
    };

    auto add_root = [&](double density,
                        double lower,
                        double upper,
                        bool at_stationary) {
        if (!std::isfinite(density)) {
            return;
        }
        const double residual = pressure_residual(density);
        if (!std::isfinite(residual) ||
            std::abs(residual) > config.pressure_tolerance * 10.0) {
            return;
        }
        for (auto& existing : result.roots) {
            if (detail::same_density(existing.density, density, config)) {
                existing.bracket_lower = std::min(existing.bracket_lower, lower);
                existing.bracket_upper = std::max(existing.bracket_upper, upper);
                existing.at_stationary_point =
                    existing.at_stationary_point || at_stationary;
                return;
            }
        }
        const double d_value = derivative(density);
        result.roots.push_back({
            density,
            residual,
            d_value,
            lower,
            upper,
            d_value > derivative_tolerance,
            at_stationary,
        });
    };

    auto refine_simple_root = [&](double density,
                                  double lower,
                                  double upper) {
        double left = lower;
        double right = upper;
        double f_left = pressure_residual(left);
        double current = std::clamp(density, left, right);
        for (int iteration = 0; iteration < 32; ++iteration) {
            const double f_current = pressure_residual(current);
            if (!std::isfinite(f_current) ||
                std::abs(f_current) <= config.pressure_tolerance) {
                return current;
            }
            const double d_current = derivative(current);
            double candidate = std::numeric_limits<double>::quiet_NaN();
            if (std::isfinite(d_current) &&
                std::abs(d_current) > derivative_tolerance) {
                candidate = current - f_current / d_current;
            }
            if (!std::isfinite(candidate) ||
                !(candidate > left && candidate < right)) {
                candidate = 0.5 * (left + right);
            }
            const double f_candidate = pressure_residual(candidate);
            if (!std::isfinite(f_candidate)) {
                return current;
            }
            if ((f_left <= 0.0 && f_candidate >= 0.0) ||
                (f_left >= 0.0 && f_candidate <= 0.0)) {
                right = candidate;
            } else {
                left = candidate;
                f_left = f_candidate;
            }
            if (candidate == current) {
                return current;
            }
            current = candidate;
        }
        return current;
    };

    for (double boundary : boundaries) {
        const double residual = pressure_residual(boundary);
        if (std::isfinite(residual) &&
            std::abs(residual) <= config.pressure_tolerance) {
            add_root(
                boundary,
                boundary,
                boundary,
                is_stationary_density(boundary));
        }
    }

    for (std::size_t index = 1; index < boundaries.size(); ++index) {
        const double lower = boundaries[index - 1];
        const double upper = boundaries[index];
        const double f_lower = pressure_residual(lower);
        const double f_upper = pressure_residual(upper);
        if (!std::isfinite(f_lower) || !std::isfinite(f_upper)) {
            continue;
        }
        if (f_lower * f_upper < 0.0) {
            const double bracketed_density = detail::bisect_sign_change(
                pressure_residual,
                lower,
                upper,
                f_lower,
                f_upper,
                config.pressure_tolerance,
                config);
            const double density = refine_simple_root(
                bracketed_density, lower, upper);
            add_root(density, lower, upper, false);
        }
    }

    std::sort(
        result.roots.begin(),
        result.roots.end(),
        [](const Root& lhs, const Root& rhs) {
            return lhs.density < rhs.density;
        });

    result.success = !result.invalid_evaluation;
    return result;
}

}  // namespace stationary_roots
