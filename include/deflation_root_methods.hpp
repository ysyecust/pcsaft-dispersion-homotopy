#pragma once
// ============================================================
// deflation_root_methods.hpp
//
// "Find-and-hide" density-root enumeration by deflation, the second
// reading of Monroy-Loperena, Chem. Thermodyn. Therm. Anal. 22 (2026)
// 100297.  A root that has been found is hidden from later searches by
// dividing it out of the residual:
//
//   G(rho) = F(rho) / prod_i (rho - rho_i),   F(rho) = P(rho) - P_target
//
// Newton on G needs no extra model evaluation, because
//
//   G / G' = F / ( F' - F * sum_i 1 / (rho - rho_i) ).
//
// The partition reading of the same reference -- locate the isotherm
// extrema, then bracket roots inside the resulting monotonic
// subintervals -- is already covered by direct_roots::stationary_partition.
//
// Both readings share the property this work contrasts against: the
// enumeration terminates on a heuristic (no new root within the restart
// budget), so a returned set carries no completeness certificate.
// ============================================================

#include "direct_root_methods.hpp"
#include "stationary_root_solver.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace deflation_roots {

struct DeflationConfig {
    int restart_count = 6;      // packing-fraction spaced restarts per sweep
    int maximum_sweeps = 8;     // sweeps without a new root end the search
    int maximum_iterations = 80;
    double damping_fraction = 0.5;  // limit on a single relative step
};

// One deflated Newton solve from a single start.  Returns true when a new
// root, distinct from every hidden root, has been isolated.
inline bool deflated_newton(
    const stationary_roots::Equation& equation,
    double initial_density,
    const std::vector<double>& hidden,
    const direct_roots::MethodConfig& config,
    const DeflationConfig& deflation,
    direct_roots::MethodResult& result,
    double& found_density,
    double& found_residual,
    double& found_derivative) {
    const double lower = equation.density_min;
    const double upper = equation.density_max;
    const double margin = 16.0 * std::numeric_limits<double>::epsilon() *
        std::max(1.0, upper);
    if (!std::isfinite(initial_density) || lower >= upper) {
        return false;
    }
    double density = std::clamp(
        initial_density, lower + margin, upper - margin);

    for (int iteration = 0;
         iteration < deflation.maximum_iterations;
         ++iteration) {
        const double residual =
            equation.pressure(density) - equation.target_pressure;
        const double derivative = equation.derivative(density);
        ++result.pressure_evaluations;
        ++result.derivative_evaluations;
        if (!std::isfinite(residual) || !std::isfinite(derivative)) {
            return false;
        }
        if (std::abs(residual) <= config.pressure_tolerance) {
            // Reject a return to an already hidden root.
            for (double hidden_density : hidden) {
                if (direct_roots::detail::same_density(
                        hidden_density, density, config)) {
                    return false;
                }
            }
            found_density = density;
            found_residual = residual;
            found_derivative = derivative;
            return true;
        }

        // Deflated Newton denominator: F' - F * sum 1 / (rho - rho_i).
        double deflation_sum = 0.0;
        bool on_hidden_root = false;
        for (double hidden_density : hidden) {
            const double gap = density - hidden_density;
            if (std::abs(gap) < config.density_tolerance *
                    std::max(1.0, std::abs(density))) {
                on_hidden_root = true;
                break;
            }
            deflation_sum += 1.0 / gap;
        }
        if (on_hidden_root) {
            return false;
        }
        const double denominator = derivative - residual * deflation_sum;
        if (!std::isfinite(denominator) ||
            std::abs(denominator) <= config.derivative_tolerance) {
            return false;
        }

        double step = residual / denominator;
        const double limit = deflation.damping_fraction *
            std::max(std::abs(density), 1e-12);
        if (std::abs(step) > limit) {
            step = step > 0.0 ? limit : -limit;
        }
        double candidate = density - step;
        if (!std::isfinite(candidate)) {
            return false;
        }
        candidate = std::clamp(candidate, lower + margin, upper - margin);
        if (std::abs(candidate - density) <=
            config.density_tolerance * std::max(1.0, std::abs(density))) {
            return false;
        }
        density = candidate;
    }
    return false;
}

// Find-and-hide enumeration.  Restarts are placed on the same
// packing-fraction grid used by the multi-start Newton baselines, so the
// two differ only by the deflation step.
inline direct_roots::MethodResult find_and_hide(
    const stationary_roots::Equation& equation,
    const direct_roots::MethodConfig& config = {},
    const DeflationConfig& deflation = {}) {
    direct_roots::MethodResult result;
    const double lower = equation.density_min;
    const double upper = equation.density_max;
    if (!(lower < upper)) {
        result.failure_reason = "invalid_domain";
        return result;
    }

    std::vector<double> hidden;
    for (int sweep = 0; sweep < deflation.maximum_sweeps; ++sweep) {
        bool found_in_sweep = false;
        for (int start = 0; start < deflation.restart_count; ++start) {
            const double fraction = deflation.restart_count <= 1
                ? 0.5
                : static_cast<double>(start) /
                      static_cast<double>(deflation.restart_count - 1);
            const double initial = lower + fraction * (upper - lower);
            double density = 0.0;
            double residual = 0.0;
            double derivative = 0.0;
            if (!deflated_newton(
                    equation, initial, hidden, config, deflation,
                    result, density, residual, derivative)) {
                continue;
            }
            const std::size_t before = result.roots.size();
            direct_roots::detail::add_root(
                result, density, residual, derivative, config);
            if (result.roots.size() != before) {
                hidden.push_back(density);
                found_in_sweep = true;
            }
        }
        if (!found_in_sweep) {
            break;
        }
    }

    direct_roots::detail::finish(result);
    return result;
}

}  // namespace deflation_roots
