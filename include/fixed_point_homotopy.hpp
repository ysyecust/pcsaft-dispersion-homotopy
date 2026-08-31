#pragma once
// ============================================================
// fixed_point_homotopy.hpp
//
// Global fixed-point homotopy baseline in the sense of
// Aslam and Sunol, Ind. Eng. Chem. Res. 45 (2006) 3303-3310,
// "Reliable computation of all the density roots of the SAFT
// equation of state through global fixed-point homotopy".
//
// Embedding (non-dimensional in density, pressure-scaled):
//
//   H(rho, lambda) = lambda * [P(rho) - P_target]
//                  + (1 - lambda) * P_scale * (rbar - rbar_0)
//
//   rbar = rho / rho_ref,   P_scale = pressure_reference(...)
//
//   lambda = 0 : H = P_scale * (rbar - rbar_0), unique root rbar_0
//   lambda = 1 : H = P(rho) - P_target, the target PC-SAFT equation
//
// The embedding is expressed as an EoSHomotopyInterface so that the
// SAME predictor-corrector, arclength step control, lambda = 1 event
// detection, root refinement, fold handling and failure taxonomy used
// by the dispersion-strength solver also trace this path.  The
// comparison therefore isolates the embedding, not the continuation
// implementation.
//
// The reference method does NOT choose its anchor freely.  Its central device
// is a starting-point criterion (their Eq. 3.4): letting t -> infinity in the
// homotopy gives
//
//     F(x) - x + x_0 = 0,
//
// and x_0 is selected so that this equation has the minimum number of real
// roots, which keeps the homotopy parameter bounded and places all real roots
// on one path.  In the notation used here the roots of their Eq. 3.4 are
// exactly the poles of the fixed-point solution graph, so the criterion is a
// pole-avoidance rule.  select_anchor_by_criterion implements it.
//
// One formulation difference must be kept in view.  Aslam and Sunol work with
// the polynomial form of the SAFT compressibility equation (seventh degree for
// a pure component, ninth for a mixture), which is defined for every real
// value of the variable, and their published anchors lie far outside the
// physical range: x_0 = 15.0 for nitrogen and -28.0 for carbon dioxide, while
// the reduced density itself cannot exceed 0.7405.  The rational PC-SAFT
// pressure used here is singular at eta = 1, so a path cannot be started
// outside the admissible interval and returned to it.  The criterion is
// therefore applied over the admissible interval, and the minimum root count
// it can actually achieve there is reported alongside the result.
// ============================================================

#include "complete_homotopy_curve.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace fixed_point_homotopy {

// Choice of the fixed-point anchor rbar_0.
enum class AnchorRule {
    ideal_gas,   // rho_0 = P_target / (R T), clamped to the domain
    mid_domain,  // rho_0 = midpoint of the admissible density interval
    hard_chain,  // rho_0 = the hard-chain root used by the proposed method
    multi_start, // union over several anchors spaced in packing fraction
    criterion,   // Aslam and Sunol starting-point criterion (their Eq. 3.4)
};

inline std::string anchor_rule_name(AnchorRule rule) {
    switch (rule) {
        case AnchorRule::ideal_gas: return "ideal_gas";
        case AnchorRule::mid_domain: return "mid_domain";
        case AnchorRule::hard_chain: return "hard_chain";
        case AnchorRule::multi_start: return "multi_start";
        case AnchorRule::criterion: return "criterion";
    }
    return "unknown";
}

struct Result {
    bool anchor_solved = false;
    double anchor_rho = std::numeric_limits<double>::quiet_NaN();
    bool complete = false;
    int fold_count = 0;
    int total_state_evaluations = 0;
    double minimum_lambda = std::numeric_limits<double>::infinity();
    double maximum_lambda = -std::numeric_limits<double>::infinity();
    homotopy::CompleteCurveFailure failure =
        homotopy::CompleteCurveFailure::none;
    std::vector<homotopy::RootEvent> roots;
};

// Build the fixed-point homotopy family as an EoSHomotopyInterface.
//
// Every residual evaluation of the returned interface performs exactly one
// underlying PC-SAFT state evaluation, matching the accounting granularity
// used for the dispersion-strength solver.
inline homotopy::EoSHomotopyInterface embed(
    const homotopy::EoSHomotopyInterface& eos,
    double pressure_target,
    double rbar_anchor) {
    const double pressure_scale =
        homotopy::complete_curve_detail::pressure_reference(
            eos, pressure_target);
    const double rho_ref = eos.rho_ref;

    homotopy::EoSHomotopyInterface embedded;
    embedded.rho_min = eos.rho_min;
    embedded.rho_max = eos.rho_max;
    embedded.rho_ref = eos.rho_ref;
    embedded.T = eos.T;
    // A generic fixed-point embedding carries no model-level certificate.
    embedded.hard_chain_anchor_is_lowest_root = false;
    embedded.dispersion_pressure_strictly_negative = false;

    const auto reference_term = [pressure_scale, rho_ref, rbar_anchor](
        double rho) {
        return pressure_scale * (rho / rho_ref - rbar_anchor);
    };

    embedded.P = [source = eos.P, pressure_target, reference_term](
        double rho, double lambda) {
        const double target_residual = source(rho, 1.0) - pressure_target;
        return pressure_target + lambda * target_residual +
            (1.0 - lambda) * reference_term(rho);
    };
    embedded.dPdrho = [source = eos.dPdrho, pressure_scale, rho_ref](
        double rho, double lambda) {
        return lambda * source(rho, 1.0) +
            (1.0 - lambda) * pressure_scale / rho_ref;
    };
    embedded.dPdlam = [source = eos.P, pressure_target, reference_term](
        double rho, double) {
        return (source(rho, 1.0) - pressure_target) - reference_term(rho);
    };
    if (eos.evaluate_all) {
        embedded.evaluate_all =
            [source = eos.evaluate_all, pressure_target, pressure_scale,
             rho_ref, reference_term](double rho, double lambda) {
                const auto value = source(rho, 1.0);
                const double target_residual = value.P - pressure_target;
                const double anchor_residual = reference_term(rho);
                homotopy::EoSEvaluation embedded_value;
                embedded_value.P = pressure_target +
                    lambda * target_residual +
                    (1.0 - lambda) * anchor_residual;
                embedded_value.dPdrho = lambda * value.dPdrho +
                    (1.0 - lambda) * pressure_scale / rho_ref;
                embedded_value.dPdlam = target_residual - anchor_residual;
                return embedded_value;
            };
    }
    return embedded;
}

// ---------------------------------------------------------------------------
// Aslam and Sunol starting-point criterion (their Eq. 3.4).
//
// Writing the fixed-point residual with the pressure scale S used here,
//
//     H = lambda [P(rho) - P_target] + (1 - lambda) S (rbar - rbar_0),
//
// the t -> infinity limit of their Eq. 3.1 gives  F(x) - x + x_0 = 0, i.e.
//
//     rbar_0 = c(rbar),      c(rbar) = rbar - [P(rho) - P_target] / S.
//
// The real roots of their Eq. 3.4 at a level rbar_0 are the crossings of the
// curve c by the horizontal line at that level, and those crossings are the
// poles of the solution graph.  The criterion selects the level crossed the
// fewest times.  Their Figures 1, 3, 5 and 8 are exactly this scan, performed
// once per state.
// ---------------------------------------------------------------------------
struct AnchorScan {
    bool success = false;
    double rbar_anchor = std::numeric_limits<double>::quiet_NaN();
    int crossing_count = -1;     // roots of Eq. 3.4 at the selected level
    int minimum_achievable = -1; // best count over all admissible levels
    int state_evaluations = 0;   // cost of the scan, charged to the method
};

inline AnchorScan select_anchor_by_criterion(
    const homotopy::EoSHomotopyInterface& eos,
    double pressure_target,
    const homotopy::CompleteCurveConfig& config,
    int mesh_intervals = 2000) {
    AnchorScan scan;
    const double lower = eos.rho_min * (1.0 + config.boundary_relative_margin);
    const double upper = eos.rho_max * (1.0 - config.boundary_relative_margin);
    if (!(lower < upper) || mesh_intervals < 8) {
        return scan;
    }
    const double pressure_scale =
        homotopy::complete_curve_detail::pressure_reference(
            eos, pressure_target);

    const int points = mesh_intervals + 1;
    std::vector<double> rbar(points);
    std::vector<double> curve(points);
    for (int index = 0; index < points; ++index) {
        const double fraction =
            static_cast<double>(index) / static_cast<double>(mesh_intervals);
        const double rho = lower + fraction * (upper - lower);
        rbar[index] = rho / eos.rho_ref;
        const double residual = eos.P(rho, 1.0) - pressure_target;
        ++scan.state_evaluations;
        if (!std::isfinite(residual)) {
            return scan;
        }
        curve[index] = rbar[index] - residual / pressure_scale;
    }

    // Crossing count as a function of level, by a sweep over segment spans.
    std::vector<std::pair<double, int>> events;
    events.reserve(2 * mesh_intervals);
    for (int index = 0; index < mesh_intervals; ++index) {
        const double a = curve[index];
        const double b = curve[index + 1];
        if (!std::isfinite(a) || !std::isfinite(b) || a == b) {
            continue;
        }
        events.emplace_back(std::min(a, b), 1);
        events.emplace_back(std::max(a, b), -1);
    }
    if (events.empty()) {
        return scan;
    }
    std::sort(events.begin(), events.end());

    // Local extrema of the curve bound the bands of equal crossing count; the
    // distance to the nearest one is used to prefer a robust level.
    std::vector<double> extrema;
    for (int index = 1; index + 1 < points; ++index) {
        const double previous = curve[index] - curve[index - 1];
        const double next = curve[index + 1] - curve[index];
        if (previous * next < 0.0) {
            extrema.push_back(curve[index]);
        }
    }

    const auto crossings_at = [&](double level) {
        int count = 0;
        for (const auto& [value, delta] : events) {
            if (value > level) {
                break;
            }
            count += delta;
        }
        return count;
    };

    // Candidate anchors are densities inside the admissible interval, because
    // a rational PC-SAFT path cannot start outside it.
    // The trace must start strictly inside the admissible interval, so the
    // candidate levels exclude a small band at each end.
    const int first = std::max(1, points / 50);
    const int last = std::min(points - 2, points - 1 - points / 50);
    int best_count = std::numeric_limits<int>::max();
    double best_margin = -1.0;
    double best_level = std::numeric_limits<double>::quiet_NaN();
    int minimum_count = std::numeric_limits<int>::max();
    for (int index = first; index <= last; ++index) {
        const double level = rbar[index];
        const int count = crossings_at(level);
        minimum_count = std::min(minimum_count, count);
        double margin = std::numeric_limits<double>::infinity();
        for (double extremum : extrema) {
            margin = std::min(margin, std::abs(level - extremum));
        }
        if (count < best_count ||
            (count == best_count && margin > best_margin)) {
            best_count = count;
            best_margin = margin;
            best_level = level;
        }
    }
    if (!std::isfinite(best_level)) {
        return scan;
    }
    scan.success = true;
    scan.rbar_anchor = best_level;
    scan.crossing_count = best_count;
    scan.minimum_achievable = minimum_count;
    return scan;
}

// Select rbar_0 according to the requested rule.  The hard-chain rule reuses
// the anchor solver of the proposed method so that a head-to-head run differs
// only in the embedding.
inline bool select_anchor(
    const homotopy::EoSHomotopyInterface& eos,
    double pressure_target,
    AnchorRule rule,
    const homotopy::CompleteCurveConfig& config,
    double& rbar_anchor,
    int& state_evaluations) {
    constexpr double gas_constant = 8.31446261815324;
    const double lower = eos.rho_min * (1.0 + config.boundary_relative_margin);
    const double upper = eos.rho_max * (1.0 - config.boundary_relative_margin);
    state_evaluations = 0;

    double rho_anchor = std::numeric_limits<double>::quiet_NaN();
    if (rule == AnchorRule::criterion) {
        const auto scan = select_anchor_by_criterion(
            eos, pressure_target, config);
        state_evaluations = scan.state_evaluations;
        if (!scan.success) {
            return false;
        }
        rbar_anchor = scan.rbar_anchor;
        return true;
    }
    if (rule == AnchorRule::ideal_gas) {
        rho_anchor = pressure_target /
            std::max(1e-30, gas_constant * eos.T);
    } else if (rule == AnchorRule::mid_domain) {
        rho_anchor = 0.5 * (lower + upper);
    } else {
        const auto anchor =
            homotopy::complete_curve_detail::solve_hard_chain_anchor(
                eos, pressure_target, config);
        state_evaluations = anchor.state_evaluations;
        if (!anchor.success) {
            return false;
        }
        rho_anchor = anchor.rho;
    }
    if (!std::isfinite(rho_anchor)) {
        return false;
    }
    rho_anchor = std::clamp(rho_anchor, lower, upper);
    if (!(rho_anchor > lower && rho_anchor < upper)) {
        return false;
    }
    rbar_anchor = rho_anchor / eos.rho_ref;
    return true;
}

// Trace the complete fixed-point homotopy curve and collect every
// lambda = 1 intersection.  Both density directions are always traced
// because the embedding supplies no low-density certificate.
inline Result solve(
    const homotopy::EoSHomotopyInterface& eos,
    double pressure_target,
    AnchorRule rule,
    const homotopy::CompleteCurveConfig& config = {}) {
    Result result;
    double rbar_anchor = 0.0;
    int anchor_evaluations = 0;
    if (!select_anchor(
            eos, pressure_target, rule, config,
            rbar_anchor, anchor_evaluations)) {
        result.failure = homotopy::CompleteCurveFailure::invalid_anchor;
        result.total_state_evaluations = anchor_evaluations;
        return result;
    }

    const auto embedded = embed(eos, pressure_target, rbar_anchor);
    const auto traced = homotopy::trace_complete_curve(
        embedded, pressure_target, rbar_anchor, config);

    result.anchor_solved = true;
    result.anchor_rho = rbar_anchor * eos.rho_ref;
    result.complete = traced.complete;
    result.fold_count = traced.fold_count;
    result.total_state_evaluations =
        traced.total_state_evaluations + anchor_evaluations;
    result.minimum_lambda = traced.minimum_lambda;
    result.maximum_lambda = traced.maximum_lambda;
    result.roots = traced.roots;
    result.failure = traced.complete
        ? homotopy::CompleteCurveFailure::none
        : (traced.higher_density.failure !=
               homotopy::CompleteCurveFailure::none
           ? traced.higher_density.failure
           : traced.lower_density.failure);
    return result;
}

// Multi-anchor variant.  A single fixed-point path is bounded by the poles of
// its own graph, so the reference method is given its strongest fair form
// here: several anchors are traced and every lambda = 1 intersection found by
// any of them is retained.  Roots are merged on the same relative density
// tolerance used inside a single trace.
//
// Like the multi-start Newton baselines, the union carries no completeness
// certificate: the search stops when the anchor list is exhausted, not when
// the admissible domain has been shown to hold no further root.
inline Result solve_multi_start(
    const homotopy::EoSHomotopyInterface& eos,
    double pressure_target,
    int anchor_count,
    const homotopy::CompleteCurveConfig& config = {},
    bool logarithmic_spacing = false) {
    Result result;
    const double lower = eos.rho_min * (1.0 + config.boundary_relative_margin);
    const double upper = eos.rho_max * (1.0 - config.boundary_relative_margin);
    const double merge_tolerance =
        config.root_merge_relative_tolerance;

    bool any_anchor = false;
    bool any_complete = false;
    for (int index = 0; index < std::max(anchor_count, 1); ++index) {
        const double fraction = anchor_count <= 1
            ? 0.5
            : static_cast<double>(index) /
                  static_cast<double>(anchor_count - 1);
        // The admissible interval spans several orders of magnitude, so
        // logarithmic placement is the appropriate generous choice; uniform
        // placement leaves the vapor region without an anchor.
        const double rho_anchor = logarithmic_spacing
            ? std::clamp(
                  std::exp(std::log(lower) +
                           fraction * (std::log(upper) - std::log(lower))),
                  lower, upper)
            : std::clamp(lower + fraction * (upper - lower), lower, upper);
        if (!(rho_anchor > lower && rho_anchor < upper)) {
            continue;
        }
        const double rbar_anchor = rho_anchor / eos.rho_ref;
        const auto embedded = embed(eos, pressure_target, rbar_anchor);
        const auto traced = homotopy::trace_complete_curve(
            embedded, pressure_target, rbar_anchor, config);
        any_anchor = true;
        any_complete = any_complete || traced.complete;
        result.total_state_evaluations += traced.total_state_evaluations;
        result.fold_count += traced.fold_count;
        for (const auto& candidate : traced.roots) {
            bool duplicate = false;
            for (const auto& existing : result.roots) {
                if (std::abs(existing.rho - candidate.rho) <=
                    merge_tolerance *
                        std::max({1.0, std::abs(existing.rho),
                                  std::abs(candidate.rho)})) {
                    duplicate = true;
                    break;
                }
            }
            if (!duplicate) {
                result.roots.push_back(candidate);
            }
        }
    }
    std::sort(
        result.roots.begin(), result.roots.end(),
        [](const homotopy::RootEvent& lhs, const homotopy::RootEvent& rhs) {
            return lhs.rho < rhs.rho;
        });
    result.anchor_solved = any_anchor;
    result.complete = any_complete;
    result.failure = any_anchor
        ? homotopy::CompleteCurveFailure::none
        : homotopy::CompleteCurveFailure::invalid_anchor;
    return result;
}

}  // namespace fixed_point_homotopy
