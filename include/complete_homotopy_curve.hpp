#pragma once

#include "homotopy_solver.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>
#include <vector>

namespace homotopy {

enum class CompleteCurveFailure {
    none,
    invalid_input,
    invalid_anchor,
    singular_curve,
    corrector_failure,
    step_below_minimum,
    maximum_steps,
    evaluation_budget,
    auxiliary_parameter_limit,
    domain_exit,
};

inline const char* to_string(CompleteCurveFailure failure) {
    switch (failure) {
        case CompleteCurveFailure::none: return "none";
        case CompleteCurveFailure::invalid_input: return "invalid_input";
        case CompleteCurveFailure::invalid_anchor: return "invalid_anchor";
        case CompleteCurveFailure::singular_curve: return "singular_curve";
        case CompleteCurveFailure::corrector_failure: return "corrector_failure";
        case CompleteCurveFailure::step_below_minimum: return "step_below_minimum";
        case CompleteCurveFailure::maximum_steps: return "maximum_steps";
        case CompleteCurveFailure::evaluation_budget: return "evaluation_budget";
        case CompleteCurveFailure::auxiliary_parameter_limit:
            return "auxiliary_parameter_limit";
        case CompleteCurveFailure::domain_exit: return "domain_exit";
    }
    return "unknown";
}

struct CompleteCurveConfig {
    double ds_init = 0.02;
    double ds_min = 1e-10;
    double ds_max = 0.10;
    double residual_tolerance = 1e-10;
    double event_tolerance = 1e-9;
    double tangent_lambda_tolerance = 1e-8;
    double root_merge_relative_tolerance = 1e-8;
    double tangent_root_merge_relative_tolerance = 1e-5;
    double event_maximum_rbar_span = 0.16;
    double event_near_target_rbar_span = 1e-4;
    double event_near_target_lambda_guard = 0.0001;
    double boundary_relative_margin = 1e-10;
    double tangent_angle_target = 0.20;
    double maximum_absolute_lambda = 1e6;
    int max_steps_per_direction = 10000;
    int max_corrector_iterations = 20;
    int max_event_iterations = 80;
    int max_evaluations = std::numeric_limits<int>::max();
    bool record_path = false;
};

struct CompleteCurvePoint {
    double rbar = 0.0;
    double lambda = 0.0;
    double pressure_residual = 0.0;
    double drbar_ds = 0.0;
    double dlambda_ds = 0.0;
    double ds = 0.0;
};

struct RootEvent {
    double rho = 0.0;
    double pressure_residual = 0.0;
    double dPdrho = 0.0;
    bool stable = false;
    bool tangent = false;
    int trace_direction = 0;
};

struct DirectionTrace {
    int density_direction = 0;
    bool reached_density_boundary = false;
    bool no_target_root_certified = false;
    CompleteCurveFailure failure = CompleteCurveFailure::none;
    int accepted_steps = 0;
    int rejected_steps = 0;
    int state_evaluations = 0;
    int fold_count = 0;
    double minimum_lambda = std::numeric_limits<double>::infinity();
    double maximum_lambda = -std::numeric_limits<double>::infinity();
    std::vector<CompleteCurvePoint> path;
    std::vector<RootEvent> roots;
};

struct CompleteCurveResult {
    bool anchor_solved = false;
    double anchor_rho = std::numeric_limits<double>::quiet_NaN();
    double anchor_pressure_residual =
        std::numeric_limits<double>::infinity();
    int anchor_state_evaluations = 0;
    CompleteCurveFailure anchor_failure = CompleteCurveFailure::none;
    bool complete = false;
    DirectionTrace lower_density;
    DirectionTrace higher_density;
    std::vector<RootEvent> roots;
    int fold_count = 0;
    int total_state_evaluations = 0;
    double minimum_lambda = std::numeric_limits<double>::infinity();
    double maximum_lambda = -std::numeric_limits<double>::infinity();
    bool used_density_scan = false;
    bool used_newton_rescue = false;
};

namespace complete_curve_detail {

struct Evaluator {
    const EoSHomotopyInterface& eos;
    double pressure_target;
    double pressure_reference;
    int maximum_evaluations;
    int count = 0;
    bool exhausted = false;

    NDResidual operator()(double rbar, double lambda) {
        if (count >= maximum_evaluations) {
            exhausted = true;
            const double nan = std::numeric_limits<double>::quiet_NaN();
            return {nan, nan, nan};
        }
        ++count;
        return eval_nd(
            eos, rbar, lambda, pressure_target, pressure_reference);
    }
};

inline double pressure_reference(
    const EoSHomotopyInterface& eos,
    double pressure_target) {
    constexpr double gas_constant = 8.31446261815324;
    return std::max({
        1.0,
        std::abs(pressure_target),
        0.01 * eos.rho_ref * gas_constant * eos.T,
    });
}

inline double pressure_tolerance(
    const EoSHomotopyInterface& eos,
    double pressure_target,
    const CompleteCurveConfig& config) {
    (void)config;
    return std::max({
        1e-12,
        1e-12 * pressure_reference(eos, pressure_target),
        1e-12 * std::max(1.0, std::abs(pressure_target)),
    });
}

inline double pressure_tolerance_at_density(
    const EoSHomotopyInterface& eos,
    double pressure_target,
    double rho,
    const CompleteCurveConfig& config) {
    const double pressure_hard_chain = eos.P(rho, 0.0);
    const double pressure_full = eos.P(rho, 1.0);
    if (!std::isfinite(pressure_hard_chain) ||
        !std::isfinite(pressure_full)) {
        return pressure_tolerance(eos, pressure_target, config);
    }
    const double cancellation_scale =
        std::abs(pressure_hard_chain) +
        std::abs(pressure_full - pressure_hard_chain) +
        std::abs(pressure_target) + 1.0;
    const double roundoff_floor =
        64.0 * std::numeric_limits<double>::epsilon() *
        cancellation_scale;
    return std::max(
        pressure_tolerance(eos, pressure_target, config),
        roundoff_floor);
}

inline bool finite_residual(const NDResidual& value) {
    return std::isfinite(value.H) && std::isfinite(value.H_rbar) &&
           std::isfinite(value.H_lam);
}

inline bool refine_target_root(
    const EoSHomotopyInterface& eos,
    double pressure_target,
    double rho_a,
    double rho_b,
    const CompleteCurveConfig& config,
    RootEvent& event,
    int direction) {
    double lower = std::min(rho_a, rho_b);
    double upper = std::max(rho_a, rho_b);
    auto residual = [&](double rho) {
        return eos.P(rho, 1.0) - pressure_target;
    };
    auto accepted_pressure_residual = [&](double rho) {
        return pressure_tolerance_at_density(
            eos, pressure_target, rho, config);
    };

    double f_lower = residual(lower);
    double f_upper = residual(upper);
    if (!std::isfinite(f_lower) || !std::isfinite(f_upper)) {
        return false;
    }

    double rho = std::abs(f_lower) <= std::abs(f_upper) ? lower : upper;
    if (f_lower * f_upper > 0.0) {
        const double lambda_a =
            (pressure_target - eos.P(lower, 0.0)) / eos.dPdlam(lower, 0.0);
        const double lambda_b =
            (pressure_target - eos.P(upper, 0.0)) / eos.dPdlam(upper, 0.0);
        const double denominator = lambda_b - lambda_a;
        if (!std::isfinite(lambda_a) || !std::isfinite(lambda_b) ||
            std::abs(denominator) < 1e-30) {
            return false;
        }
        rho = lower + (1.0 - lambda_a) * (upper - lower) / denominator;
        rho = std::clamp(rho, lower, upper);
    } else {
        for (int iteration = 0;
             iteration < config.max_event_iterations;
             ++iteration) {
            const double candidate = 0.5 * (lower + upper);

            const double f_candidate = residual(candidate);
            if (!std::isfinite(f_candidate)) {
                return false;
            }
            rho = candidate;
            if (std::abs(f_candidate) <=
                    accepted_pressure_residual(candidate) ||
                upper - lower <= config.event_tolerance *
                    std::max(1.0, std::abs(rho))) {
                break;
            }
            if (f_lower * f_candidate <= 0.0) {
                upper = candidate;
                f_upper = f_candidate;
            } else {
                lower = candidate;
                f_lower = f_candidate;
            }
        }
    }

    for (int iteration = 0; iteration < 12; ++iteration) {
        const double f = residual(rho);
        const double derivative = eos.dPdrho(rho, 1.0);
        if (!std::isfinite(f) || !std::isfinite(derivative)) {
            return false;
        }
        if (std::abs(f) <= accepted_pressure_residual(rho)) {
            break;
        }
        if (std::abs(derivative) < 1e-30) {
            break;
        }
        const double candidate = rho - f / derivative;
        if (candidate < lower || candidate > upper ||
            !std::isfinite(candidate)) {
            break;
        }
        rho = candidate;
    }

    event.rho = rho;
    event.pressure_residual = residual(rho);
    event.dPdrho = eos.dPdrho(rho, 1.0);
    event.stable = event.dPdrho > 0.0;
    event.tangent = false;
    event.trace_direction = direction;
    return std::isfinite(event.pressure_residual) &&
           std::abs(event.pressure_residual) <=
               10.0 * accepted_pressure_residual(rho);
}

inline bool graph_state_at_density(
    const EoSHomotopyInterface& eos,
    double pressure_target,
    double rho,
    double& lambda,
    double& derivative) {
    const double pressure_hard_chain = eos.P(rho, 0.0);
    const double dispersion_pressure =
        eos.P(rho, 1.0) - pressure_hard_chain;
    if (!std::isfinite(pressure_hard_chain) ||
        !std::isfinite(dispersion_pressure) ||
        std::abs(dispersion_pressure) < 1e-30) {
        return false;
    }
    lambda =
        (pressure_target - pressure_hard_chain) / dispersion_pressure;
    derivative = eos.dPdrho(rho, lambda);
    return std::isfinite(lambda) && std::isfinite(derivative);
}

inline void detect_tangent_event(
    const EoSHomotopyInterface& eos,
    double pressure_target,
    const CompleteCurvePoint& previous,
    const CompleteCurvePoint& current,
    const CompleteCurveConfig& config,
    int direction,
    std::vector<RootEvent>& roots) {
    if (previous.dlambda_ds * current.dlambda_ds >= 0.0) {
        return;
    }

    double lower = std::min(previous.rbar, current.rbar) * eos.rho_ref;
    double upper = std::max(previous.rbar, current.rbar) * eos.rho_ref;
    double lambda_lower = 0.0;
    double lambda_upper = 0.0;
    double derivative_lower = 0.0;
    double derivative_upper = 0.0;
    if (!graph_state_at_density(
            eos,
            pressure_target,
            lower,
            lambda_lower,
            derivative_lower) ||
        !graph_state_at_density(
            eos,
            pressure_target,
            upper,
            lambda_upper,
            derivative_upper) ||
        derivative_lower * derivative_upper > 0.0) {
        return;
    }

    double rho_fold =
        std::abs(derivative_lower) <= std::abs(derivative_upper)
        ? lower : upper;
    double lambda_fold =
        std::abs(derivative_lower) <= std::abs(derivative_upper)
        ? lambda_lower : lambda_upper;
    double derivative_fold =
        std::abs(derivative_lower) <= std::abs(derivative_upper)
        ? derivative_lower : derivative_upper;
    for (int iteration = 0;
         iteration < config.max_event_iterations;
         ++iteration) {
        const double midpoint = 0.5 * (lower + upper);
        double lambda_midpoint = 0.0;
        double derivative_midpoint = 0.0;
        if (!graph_state_at_density(
                eos,
                pressure_target,
                midpoint,
                lambda_midpoint,
                derivative_midpoint)) {
            return;
        }
        rho_fold = midpoint;
        lambda_fold = lambda_midpoint;
        derivative_fold = derivative_midpoint;
        if (std::abs(derivative_midpoint) <=
                config.residual_tolerance ||
            upper - lower <= config.event_tolerance *
                std::max(1.0, std::abs(midpoint))) {
            break;
        }
        if (derivative_lower * derivative_midpoint <= 0.0) {
            upper = midpoint;
            lambda_upper = lambda_midpoint;
            derivative_upper = derivative_midpoint;
        } else {
            lower = midpoint;
            lambda_lower = lambda_midpoint;
            derivative_lower = derivative_midpoint;
        }
    }

    if (std::abs(lambda_fold - 1.0) >
        config.tangent_lambda_tolerance) {
        return;
    }
    RootEvent event;
    event.rho = rho_fold;
    event.pressure_residual =
        eos.P(rho_fold, 1.0) - pressure_target;
    event.dPdrho = eos.dPdrho(rho_fold, 1.0);
    event.stable = event.dPdrho > 0.0;
    event.tangent = true;
    event.trace_direction = direction;
    if (std::isfinite(event.pressure_residual) &&
        std::abs(event.pressure_residual) <=
            10.0 * pressure_tolerance_at_density(
                eos, pressure_target, rho_fold, config)) {
        roots.push_back(event);
    }
}

inline bool graph_curve_point(
    const EoSHomotopyInterface& eos,
    double pressure_target,
    double rbar,
    int density_direction,
    CompleteCurvePoint& point) {
    const double rho = rbar * eos.rho_ref;
    double lambda = 0.0;
    double derivative = 0.0;
    if (!graph_state_at_density(
            eos, pressure_target, rho, lambda, derivative)) {
        return false;
    }
    const double dispersion = eos.dPdlam(rho, lambda);
    if (!std::isfinite(dispersion) || std::abs(dispersion) < 1e-30) {
        return false;
    }
    const double graph_slope_rbar =
        -derivative * eos.rho_ref / dispersion;
    const double norm = std::hypot(1.0, graph_slope_rbar);
    point = {
        rbar,
        lambda,
        0.0,
        density_direction / norm,
        density_direction * graph_slope_rbar / norm,
        0.0,
    };
    return true;
}

inline bool refine_fold_point(
    const EoSHomotopyInterface& eos,
    double pressure_target,
    const CompleteCurvePoint& first,
    const CompleteCurvePoint& second,
    const CompleteCurveConfig& config,
    int density_direction,
    CompleteCurvePoint& fold) {
    double lower = std::min(first.rbar, second.rbar);
    double upper = std::max(first.rbar, second.rbar);
    CompleteCurvePoint left;
    CompleteCurvePoint right;
    if (!graph_curve_point(
            eos, pressure_target, lower, density_direction, left) ||
        !graph_curve_point(
            eos, pressure_target, upper, density_direction, right) ||
        left.dlambda_ds * right.dlambda_ds > 0.0) {
        return false;
    }
    fold = std::abs(left.dlambda_ds) <= std::abs(right.dlambda_ds)
        ? left : right;
    for (int iteration = 0;
         iteration < config.max_event_iterations;
         ++iteration) {
        const double middle = 0.5 * (lower + upper);
        CompleteCurvePoint candidate;
        if (!graph_curve_point(
                eos, pressure_target, middle, density_direction, candidate)) {
            return false;
        }
        fold = candidate;
        if (std::abs(candidate.dlambda_ds) <= config.residual_tolerance ||
            upper - lower <= config.event_tolerance *
                std::max(1.0, std::abs(middle))) {
            return true;
        }
        if (left.dlambda_ds * candidate.dlambda_ds <= 0.0) {
            upper = middle;
            right = candidate;
        } else {
            lower = middle;
            left = candidate;
        }
    }
    return true;
}

inline void detect_crossing_event(
    const EoSHomotopyInterface& eos,
    double pressure_target,
    const CompleteCurvePoint& previous,
    const CompleteCurvePoint& current,
    const CompleteCurveConfig& config,
    int direction,
    std::vector<RootEvent>& roots);

inline bool process_event_segment(
    const EoSHomotopyInterface& eos,
    double pressure_target,
    const CompleteCurvePoint& first,
    const CompleteCurvePoint& second,
    const CompleteCurveConfig& config,
    int density_direction,
    std::vector<RootEvent>& roots,
    int& fold_count,
    int depth = 0) {
    const double first_event = first.lambda - 1.0;
    const double second_event = second.lambda - 1.0;
    const double span = std::abs(second.rbar - first.rbar);
    if (span > config.event_near_target_rbar_span && depth < 24) {
        CompleteCurvePoint middle;
        if (!graph_curve_point(
                eos,
                pressure_target,
                0.5 * (first.rbar + second.rbar),
                density_direction,
                middle)) {
            return false;
        }
        const double middle_event = middle.lambda - 1.0;
        const bool near_target = std::min({
            std::abs(first_event),
            std::abs(middle_event),
            std::abs(second_event),
        }) <= config.event_near_target_lambda_guard;
        const bool midpoint_reveals_crossing =
            first_event * middle_event <= 0.0 ||
            middle_event * second_event <= 0.0;
        const bool midpoint_reveals_fold =
            first.dlambda_ds * middle.dlambda_ds < 0.0 ||
            middle.dlambda_ds * second.dlambda_ds < 0.0;
        if (span > config.event_maximum_rbar_span || near_target ||
            midpoint_reveals_crossing || midpoint_reveals_fold) {
            return process_event_segment(
                       eos, pressure_target, first, middle, config,
                       density_direction, roots, fold_count, depth + 1) &&
                   process_event_segment(
                       eos, pressure_target, middle, second, config,
                       density_direction, roots, fold_count, depth + 1);
        }
    }

    if (first_event * second_event <= 0.0) {
        detect_crossing_event(
            eos, pressure_target, first, second, config,
            density_direction, roots);
    }

    if (first.dlambda_ds * second.dlambda_ds < 0.0) {
        CompleteCurvePoint graph_first;
        CompleteCurvePoint graph_second;
        if (!graph_curve_point(
                eos,
                pressure_target,
                first.rbar,
                density_direction,
                graph_first) ||
            !graph_curve_point(
                eos,
                pressure_target,
                second.rbar,
                density_direction,
                graph_second)) {
            return false;
        }
        if (graph_first.dlambda_ds * graph_second.dlambda_ds >= 0.0) {
            return true;
        }
        CompleteCurvePoint fold;
        if (!refine_fold_point(
                eos, pressure_target, graph_first, graph_second, config,
                density_direction, fold)) {
            return false;
        }
        const double target_model_fold_residual =
            eos.P(fold.rbar * eos.rho_ref, 1.0) - pressure_target;
        if (std::isfinite(target_model_fold_residual) &&
            std::abs(target_model_fold_residual) <=
                pressure_tolerance_at_density(
                    eos,
                    pressure_target,
                    fold.rbar * eos.rho_ref,
                    config)) {
            detect_tangent_event(
                eos, pressure_target, graph_first, graph_second, config,
                density_direction, roots);
        } else {
            detect_crossing_event(
                eos, pressure_target, graph_first, fold, config,
                density_direction, roots);
            detect_crossing_event(
                eos, pressure_target, fold, graph_second, config,
                density_direction, roots);
        }
        ++fold_count;
        return true;
    }

    if (first_event * second_event > 0.0) {
        detect_crossing_event(
            eos, pressure_target, first, second, config,
            density_direction, roots);
    }
    return true;
}

inline void detect_crossing_event(
    const EoSHomotopyInterface& eos,
    double pressure_target,
    const CompleteCurvePoint& previous,
    const CompleteCurvePoint& current,
    const CompleteCurveConfig& config,
    int direction,
    std::vector<RootEvent>& roots) {
    const double previous_event = previous.lambda - 1.0;
    const double current_event = current.lambda - 1.0;
    const bool endpoint_event =
        std::abs(previous_event) <= config.event_tolerance ||
        std::abs(current_event) <= config.event_tolerance;
    if (!endpoint_event && previous_event * current_event > 0.0) {
        return;
    }

    RootEvent event;
    if (refine_target_root(
            eos,
            pressure_target,
            previous.rbar * eos.rho_ref,
            current.rbar * eos.rho_ref,
            config,
            event,
            direction)) {
        roots.push_back(event);
    }
}

inline bool correct_at_fixed_density(
    Evaluator& evaluate,
    double rbar,
    double lambda_initial,
    const CompleteCurveConfig& config,
    double& lambda) {
    lambda = lambda_initial;
    for (int iteration = 0;
         iteration < config.max_corrector_iterations;
         ++iteration) {
        const auto value = evaluate(rbar, lambda);
        if (evaluate.exhausted || !finite_residual(value)) {
            return false;
        }
        if (std::abs(value.H) <= config.residual_tolerance) {
            return true;
        }
        if (std::abs(value.H_lam) < 1e-30) {
            return false;
        }
        const double delta = value.H / value.H_lam;
        lambda -= std::clamp(delta, -1.0, 1.0);
        if (std::abs(lambda) > config.maximum_absolute_lambda) {
            return false;
        }
    }
    const auto final_value = evaluate(rbar, lambda);
    return !evaluate.exhausted && finite_residual(final_value) &&
           std::abs(final_value.H) <= config.residual_tolerance;
}

inline double auxiliary_from_lambda(double lambda, double scale) {
    return std::asinh(lambda / scale);
}

inline double lambda_from_auxiliary(double auxiliary, double scale) {
    return scale * std::sinh(auxiliary);
}

inline double dlambda_dauxiliary(double auxiliary, double scale) {
    return scale * std::cosh(auxiliary);
}

inline DirectionTrace trace_direction(
    const EoSHomotopyInterface& eos,
    double pressure_target,
    double rbar_anchor,
    int density_direction,
    const CompleteCurveConfig& config) {
    DirectionTrace trace;
    trace.density_direction = density_direction;

    const double reference = pressure_reference(eos, pressure_target);
    Evaluator evaluate{
        eos, pressure_target, reference, config.max_evaluations};

    const double rbar_lower =
        eos.rho_min / eos.rho_ref *
        (1.0 + config.boundary_relative_margin);
    const double rbar_upper =
        eos.rho_max / eos.rho_ref *
        (1.0 - config.boundary_relative_margin);
    if (!(rbar_lower < rbar_anchor && rbar_anchor < rbar_upper)) {
        trace.failure = CompleteCurveFailure::invalid_anchor;
        return trace;
    }

    const double auxiliary_scale = 1.0;
    double rbar = rbar_anchor;
    double density_coordinate = std::log(rbar);
    double lambda = 0.0;
    double auxiliary = auxiliary_from_lambda(lambda, auxiliary_scale);
    auto value = evaluate(rbar, lambda);
    if (evaluate.exhausted) {
        trace.failure = CompleteCurveFailure::evaluation_budget;
        return trace;
    }
    if (!finite_residual(value) ||
        std::abs(value.H) > 10.0 * config.residual_tolerance) {
        trace.failure = CompleteCurveFailure::invalid_anchor;
        trace.state_evaluations = evaluate.count;
        return trace;
    }

    double h_auxiliary = value.H_lam *
        dlambda_dauxiliary(auxiliary, auxiliary_scale);
    double h_density_coordinate = value.H_rbar * rbar;
    double tangent_norm = std::hypot(
        h_auxiliary, h_density_coordinate);
    if (tangent_norm < 1e-30) {
        trace.failure = CompleteCurveFailure::singular_curve;
        trace.state_evaluations = evaluate.count;
        return trace;
    }
    double ddensity_coordinate_ds = -h_auxiliary / tangent_norm;
    double dauxiliary_ds = value.H_rbar / tangent_norm;
    dauxiliary_ds *= rbar;
    if (density_direction * ddensity_coordinate_ds < 0.0) {
        ddensity_coordinate_ds = -ddensity_coordinate_ds;
        dauxiliary_ds = -dauxiliary_ds;
    }

    double ds = config.ds_init;
    CompleteCurvePoint previous{
        rbar, lambda, value.H * reference,
        rbar * ddensity_coordinate_ds,
        dauxiliary_ds * dlambda_dauxiliary(auxiliary, auxiliary_scale),
        ds};
    trace.minimum_lambda = lambda;
    trace.maximum_lambda = lambda;
    if (config.record_path) {
        trace.path.push_back(previous);
    }

    for (int step = 0; step < config.max_steps_per_direction; ++step) {
        double density_coordinate_predictor =
            density_coordinate + ds * ddensity_coordinate_ds;
        double rbar_predictor = std::exp(density_coordinate_predictor);
        double auxiliary_predictor = auxiliary + ds * dauxiliary_ds;
        const bool crosses_boundary = density_direction < 0
            ? rbar_predictor <= rbar_lower
            : rbar_predictor >= rbar_upper;

        if (crosses_boundary) {
            const double boundary = density_direction < 0
                ? rbar_lower : rbar_upper;
            const double boundary_density_coordinate = std::log(boundary);
            const double density_fraction =
                std::abs(density_coordinate_predictor - density_coordinate) > 1e-30
                ? (boundary_density_coordinate - density_coordinate) /
                    (density_coordinate_predictor - density_coordinate)
                : 0.0;
            const double auxiliary_guess =
                auxiliary + density_fraction *
                    (auxiliary_predictor - auxiliary);
            const double lambda_guess = lambda_from_auxiliary(
                auxiliary_guess, auxiliary_scale);
            double boundary_lambda = lambda_guess;
            if (!correct_at_fixed_density(
                    evaluate,
                    boundary,
                    lambda_guess,
                    config,
                    boundary_lambda)) {
                trace.failure = evaluate.exhausted
                    ? CompleteCurveFailure::evaluation_budget
                    : CompleteCurveFailure::corrector_failure;
                break;
            }
            const auto boundary_value = evaluate(boundary, boundary_lambda);
            if (evaluate.exhausted || !finite_residual(boundary_value)) {
                trace.failure = CompleteCurveFailure::evaluation_budget;
                break;
            }
            const double boundary_auxiliary = auxiliary_from_lambda(
                boundary_lambda, auxiliary_scale);
            const double boundary_h_auxiliary = boundary_value.H_lam *
                dlambda_dauxiliary(boundary_auxiliary, auxiliary_scale);
            const double boundary_h_density =
                boundary_value.H_rbar * boundary;
            tangent_norm = std::hypot(
                boundary_h_auxiliary, boundary_h_density);
            if (tangent_norm < 1e-30) {
                trace.failure = CompleteCurveFailure::singular_curve;
                break;
            }
            double boundary_ddensity =
                -boundary_h_auxiliary / tangent_norm;
            double boundary_dauxiliary =
                boundary_h_density / tangent_norm;
            if (density_direction * boundary_ddensity < 0.0) {
                boundary_ddensity = -boundary_ddensity;
                boundary_dauxiliary = -boundary_dauxiliary;
            }
            CompleteCurvePoint current{
                boundary,
                boundary_lambda,
                boundary_value.H * reference,
                boundary * boundary_ddensity,
                boundary_dauxiliary * dlambda_dauxiliary(
                    boundary_auxiliary, auxiliary_scale),
                ds,
            };
            if (!process_event_segment(
                    eos, pressure_target, previous, current, config,
                    density_direction, trace.roots, trace.fold_count)) {
                trace.failure = CompleteCurveFailure::corrector_failure;
                break;
            }
            ++trace.accepted_steps;
            trace.minimum_lambda =
                std::min(trace.minimum_lambda, boundary_lambda);
            trace.maximum_lambda =
                std::max(trace.maximum_lambda, boundary_lambda);
            if (config.record_path) {
                trace.path.push_back(current);
            }
            trace.reached_density_boundary = true;
            trace.failure = CompleteCurveFailure::none;
            break;
        }

        double density_coordinate_corrected =
            density_coordinate_predictor;
        double rbar_corrected = std::exp(density_coordinate_corrected);
        double auxiliary_corrected = auxiliary_predictor;
        double lambda_corrected = lambda_from_auxiliary(
            auxiliary_corrected, auxiliary_scale);
        bool corrected = false;
        for (int iteration = 0;
             iteration < config.max_corrector_iterations;
             ++iteration) {
            lambda_corrected = lambda_from_auxiliary(
                auxiliary_corrected, auxiliary_scale);
            rbar_corrected = std::exp(density_coordinate_corrected);
            const auto corrected_value = evaluate(
                rbar_corrected, lambda_corrected);
            if (evaluate.exhausted || !finite_residual(corrected_value)) {
                break;
            }
            const double arclength_residual =
                (density_coordinate_corrected -
                    density_coordinate_predictor) *
                    ddensity_coordinate_ds +
                (auxiliary_corrected - auxiliary_predictor) *
                    dauxiliary_ds;
            if (std::abs(corrected_value.H) <=
                    config.residual_tolerance &&
                std::abs(arclength_residual) <=
                    config.residual_tolerance) {
                corrected = true;
                break;
            }

            const double corrected_h_auxiliary =
                corrected_value.H_lam * dlambda_dauxiliary(
                    auxiliary_corrected, auxiliary_scale);
            const double corrected_h_density =
                corrected_value.H_rbar * rbar_corrected;
            const double determinant =
                corrected_h_density * dauxiliary_ds -
                corrected_h_auxiliary * ddensity_coordinate_ds;
            const double determinant_scale = std::max({
                1e-30,
                std::abs(corrected_h_density),
                std::abs(corrected_h_auxiliary),
            });
            if (std::abs(determinant) < 1e-14 * determinant_scale) {
                break;
            }
            const double rhs_h = -corrected_value.H;
            const double rhs_s = -arclength_residual;
            double delta_density_coordinate =
                (rhs_h * dauxiliary_ds -
                 corrected_h_auxiliary * rhs_s) / determinant;
            double delta_auxiliary =
                (corrected_h_density * rhs_s -
                 rhs_h * ddensity_coordinate_ds) / determinant;
            const double correction_norm =
                std::hypot(delta_density_coordinate, delta_auxiliary);
            if (correction_norm > 0.5) {
                delta_density_coordinate *= 0.5 / correction_norm;
                delta_auxiliary *= 0.5 / correction_norm;
            }
            density_coordinate_corrected += delta_density_coordinate;
            rbar_corrected = std::exp(density_coordinate_corrected);
            auxiliary_corrected += delta_auxiliary;
            lambda_corrected = lambda_from_auxiliary(
                auxiliary_corrected, auxiliary_scale);
            if (rbar_corrected <= rbar_lower ||
                rbar_corrected >= rbar_upper ||
                !std::isfinite(lambda_corrected) ||
                std::abs(lambda_corrected) >
                    config.maximum_absolute_lambda) {
                break;
            }
        }

        if (corrected &&
            density_direction * (rbar_corrected - rbar) <= 0.0) {
            corrected = false;
        }

        if (!corrected) {
            ++trace.rejected_steps;
            ds *= 0.5;
            if (evaluate.exhausted) {
                trace.failure = CompleteCurveFailure::evaluation_budget;
                break;
            }
            if (ds < config.ds_min) {
                trace.failure = CompleteCurveFailure::step_below_minimum;
                break;
            }
            continue;
        }

        lambda_corrected = lambda_from_auxiliary(
            auxiliary_corrected, auxiliary_scale);
        rbar_corrected = std::exp(density_coordinate_corrected);
        const auto corrected_value = evaluate(
            rbar_corrected, lambda_corrected);
        if (evaluate.exhausted || !finite_residual(corrected_value)) {
            trace.failure = CompleteCurveFailure::evaluation_budget;
            break;
        }
        h_auxiliary = corrected_value.H_lam * dlambda_dauxiliary(
            auxiliary_corrected, auxiliary_scale);
        h_density_coordinate = corrected_value.H_rbar * rbar_corrected;
        tangent_norm = std::hypot(
            h_auxiliary, h_density_coordinate);
        if (tangent_norm < 1e-30) {
            trace.failure = CompleteCurveFailure::singular_curve;
            break;
        }
        double new_ddensity_coordinate_ds =
            -h_auxiliary / tangent_norm;
        double new_dauxiliary_ds =
            h_density_coordinate / tangent_norm;
        if (density_direction * new_ddensity_coordinate_ds < 0.0) {
            new_ddensity_coordinate_ds = -new_ddensity_coordinate_ds;
            new_dauxiliary_ds = -new_dauxiliary_ds;
        }

        CompleteCurvePoint current{
            rbar_corrected,
            lambda_corrected,
            corrected_value.H * reference,
            rbar_corrected * new_ddensity_coordinate_ds,
            new_dauxiliary_ds * dlambda_dauxiliary(
                auxiliary_corrected, auxiliary_scale),
            ds,
        };
        if (!process_event_segment(
                eos, pressure_target, previous, current, config,
                density_direction, trace.roots, trace.fold_count)) {
            trace.failure = CompleteCurveFailure::corrector_failure;
            break;
        }

        double cosine =
            ddensity_coordinate_ds * new_ddensity_coordinate_ds +
            dauxiliary_ds * new_dauxiliary_ds;
        cosine = std::clamp(cosine, -1.0, 1.0);
        const double angle = std::acos(cosine);
        const double factor = std::clamp(
            config.tangent_angle_target / std::max(angle, 1e-8),
            0.5,
            1.5);
        ds = std::clamp(ds * factor, config.ds_min, config.ds_max);

        rbar = rbar_corrected;
        density_coordinate = density_coordinate_corrected;
        lambda = lambda_corrected;
        auxiliary = auxiliary_corrected;
        ddensity_coordinate_ds = new_ddensity_coordinate_ds;
        dauxiliary_ds = new_dauxiliary_ds;
        previous = current;
        ++trace.accepted_steps;
        trace.minimum_lambda =
            std::min(trace.minimum_lambda, lambda);
        trace.maximum_lambda =
            std::max(trace.maximum_lambda, lambda);
        if (config.record_path) {
            trace.path.push_back(current);
        }
    }

    if (!trace.reached_density_boundary &&
        trace.failure == CompleteCurveFailure::none) {
        trace.failure = CompleteCurveFailure::maximum_steps;
    }
    trace.state_evaluations = evaluate.count;
    return trace;
}

inline bool same_root(
    const RootEvent& lhs,
    const RootEvent& rhs,
    double relative_tolerance) {
    const double relative_window = relative_tolerance *
        std::max({1.0, std::abs(lhs.rho), std::abs(rhs.rho)});
    auto density_uncertainty = [](const RootEvent& root) {
        if (!std::isfinite(root.pressure_residual) ||
            !std::isfinite(root.dPdrho) ||
            std::abs(root.dPdrho) < 1e-30) {
            return 0.0;
        }
        return std::abs(root.pressure_residual / root.dPdrho);
    };
    const double residual_window = 2.0 *
        (density_uncertainty(lhs) + density_uncertainty(rhs));
    return std::abs(lhs.rho - rhs.rho) <=
        std::max(relative_window, residual_window);
}

struct HardChainAnchor {
    bool success = false;
    double rho = std::numeric_limits<double>::quiet_NaN();
    double pressure_residual = std::numeric_limits<double>::infinity();
    int state_evaluations = 0;
    CompleteCurveFailure failure = CompleteCurveFailure::none;
};

inline HardChainAnchor solve_hard_chain_anchor(
    const EoSHomotopyInterface& eos,
    double pressure_target,
    const CompleteCurveConfig& config) {
    HardChainAnchor anchor;
    if (!(eos.rho_ref > 0.0) || !(eos.rho_min < eos.rho_max) ||
        !std::isfinite(pressure_target)) {
        anchor.failure = CompleteCurveFailure::invalid_input;
        return anchor;
    }

    double lower = eos.rho_min *
        (1.0 + config.boundary_relative_margin);
    double upper = eos.rho_max *
        (1.0 - config.boundary_relative_margin);
    const auto residual = [&](double rho) {
        return eos.P(rho, 0.0) - pressure_target;
    };
    double f_lower = residual(lower);
    double f_upper = residual(upper);
    anchor.state_evaluations += 2;
    if (!std::isfinite(f_lower) || !std::isfinite(f_upper) ||
        f_lower * f_upper > 0.0) {
        anchor.failure = CompleteCurveFailure::invalid_anchor;
        return anchor;
    }

    const double accepted_residual =
        pressure_tolerance(eos, pressure_target, config);
    constexpr double gas_constant = 8.31446261815324;
    double rho = pressure_target /
        std::max(1e-30, gas_constant * eos.T);
    rho = std::clamp(rho, lower, upper);
    for (int iteration = 0; iteration < 100; ++iteration) {
        const double f = residual(rho);
        const double derivative = eos.dPdrho(rho, 0.0);
        anchor.state_evaluations += 2;
        if (!std::isfinite(f) || !std::isfinite(derivative)) {
            anchor.failure = CompleteCurveFailure::invalid_anchor;
            return anchor;
        }
        if (std::abs(f) <= accepted_residual) {
            anchor.success = true;
            anchor.rho = rho;
            anchor.pressure_residual = f;
            anchor.failure = CompleteCurveFailure::none;
            return anchor;
        }

        if (f_lower * f <= 0.0) {
            upper = rho;
            f_upper = f;
        } else {
            lower = rho;
            f_lower = f;
        }
        double candidate = 0.5 * (lower + upper);
        if (std::abs(derivative) > 1e-30) {
            const double newton = rho - f / derivative;
            if (newton > lower && newton < upper &&
                std::isfinite(newton)) {
                candidate = newton;
            }
        }
        rho = candidate;
        if (upper - lower <= config.event_tolerance *
                std::max(1.0, std::abs(rho))) {
            const double final_residual = residual(rho);
            ++anchor.state_evaluations;
            if (std::isfinite(final_residual) &&
                std::abs(final_residual) <=
                    10.0 * accepted_residual) {
                anchor.success = true;
                anchor.rho = rho;
                anchor.pressure_residual = final_residual;
                anchor.failure = CompleteCurveFailure::none;
                return anchor;
            }
        }
    }
    anchor.failure = CompleteCurveFailure::invalid_anchor;
    return anchor;
}

}  // namespace complete_curve_detail

inline CompleteCurveResult trace_complete_curve(
    const EoSHomotopyInterface& eos,
    double pressure_target,
    double rbar_anchor,
    const CompleteCurveConfig& config = {}) {
    CompleteCurveResult result;
    result.anchor_solved = true;
    result.anchor_rho = rbar_anchor * eos.rho_ref;
    result.anchor_pressure_residual =
        eos.P(result.anchor_rho, 0.0) - pressure_target;
    if (eos.hard_chain_anchor_is_lowest_root &&
        eos.dispersion_pressure_strictly_negative) {
        result.lower_density.density_direction = -1;
        result.lower_density.no_target_root_certified = true;
        result.lower_density.failure = CompleteCurveFailure::none;
        result.lower_density.minimum_lambda = 0.0;
        result.lower_density.maximum_lambda = 0.0;
    } else {
        result.lower_density = complete_curve_detail::trace_direction(
            eos, pressure_target, rbar_anchor, -1, config);
    }
    result.higher_density = complete_curve_detail::trace_direction(
        eos, pressure_target, rbar_anchor, +1, config);

    result.complete =
        (result.lower_density.reached_density_boundary ||
         result.lower_density.no_target_root_certified) &&
        result.higher_density.reached_density_boundary &&
        result.lower_density.failure == CompleteCurveFailure::none &&
        result.higher_density.failure == CompleteCurveFailure::none;
    result.fold_count =
        result.lower_density.fold_count + result.higher_density.fold_count;
    result.total_state_evaluations =
        result.lower_density.state_evaluations +
        result.higher_density.state_evaluations;
    result.minimum_lambda = std::min(
        result.lower_density.minimum_lambda,
        result.higher_density.minimum_lambda);
    result.maximum_lambda = std::max(
        result.lower_density.maximum_lambda,
        result.higher_density.maximum_lambda);

    result.roots = result.lower_density.roots;
    result.roots.insert(
        result.roots.end(),
        result.higher_density.roots.begin(),
        result.higher_density.roots.end());
    std::sort(
        result.roots.begin(),
        result.roots.end(),
        [](const RootEvent& lhs, const RootEvent& rhs) {
            return lhs.rho < rhs.rho;
        });
    std::vector<RootEvent> merged_roots;
    for (const auto& root : result.roots) {
        const double merge_tolerance =
            !merged_roots.empty() &&
                (merged_roots.back().tangent || root.tangent)
            ? std::max(
                  config.root_merge_relative_tolerance,
                  config.tangent_root_merge_relative_tolerance)
            : config.root_merge_relative_tolerance;
        if (merged_roots.empty() ||
            !complete_curve_detail::same_root(
                merged_roots.back(),
                root,
                merge_tolerance)) {
            merged_roots.push_back(root);
            continue;
        }
        if (root.tangent) {
            merged_roots.back() = root;
        }
    }
    result.roots = std::move(merged_roots);
    return result;
}

inline CompleteCurveResult solve_complete_density_roots(
    const EoSHomotopyInterface& eos,
    double pressure_target,
    const CompleteCurveConfig& config = {}) {
    const auto anchor = complete_curve_detail::solve_hard_chain_anchor(
        eos, pressure_target, config);
    if (!anchor.success) {
        CompleteCurveResult result;
        result.anchor_solved = false;
        result.anchor_failure = anchor.failure;
        result.anchor_state_evaluations = anchor.state_evaluations;
        result.total_state_evaluations = anchor.state_evaluations;
        return result;
    }

    auto result = trace_complete_curve(
        eos,
        pressure_target,
        anchor.rho / eos.rho_ref,
        config);
    result.anchor_solved = true;
    result.anchor_rho = anchor.rho;
    result.anchor_pressure_residual = anchor.pressure_residual;
    result.anchor_state_evaluations = anchor.state_evaluations;
    result.anchor_failure = CompleteCurveFailure::none;
    result.total_state_evaluations += anchor.state_evaluations;
    return result;
}

}  // namespace homotopy
