#pragma once

#include <algorithm>
#include <cmath>
#include <vector>

namespace solver_benchmark {

struct Root {
    double density = 0.0;
    double residual = 0.0;
    double derivative = 0.0;
    bool mechanically_stable = false;
    bool tangent = false;
    // Dimensionless mechanical-stability indicator
    //     chi = (rho / P_scale) dP/drho,
    // used so that reference and returned roots can be classified by the
    // same explicit threshold instead of by the sign of a raw derivative.
    double chi = 0.0;
};

// Classification of one root at an explicit dimensionless threshold.
enum class Stability { stable, unstable, marginal };

inline Stability classify(const Root& root, double chi_tolerance) {
    if (root.chi > chi_tolerance) {
        return Stability::stable;
    }
    if (root.chi < -chi_tolerance) {
        return Stability::unstable;
    }
    return Stability::marginal;
}

struct RootComparison {
    int matched = 0;
    int missed = 0;
    int extra = 0;
    int missed_stable = 0;
    int extra_stable = 0;
    int matched_tangent = 0;
    int missed_tangent = 0;
    bool complete_all = false;
    bool complete_stable = false;
    bool complete_tangent = false;
};

inline bool same_density(double lhs, double rhs, double relative_tolerance) {
    return std::abs(lhs - rhs) <= relative_tolerance *
        std::max({1.0, std::abs(lhs), std::abs(rhs)});
}

inline double density_uncertainty(const Root& root) {
    if (!std::isfinite(root.residual) || !std::isfinite(root.derivative) ||
        std::abs(root.derivative) < 1e-30) {
        return 0.0;
    }
    return std::abs(root.residual / root.derivative);
}

inline bool same_root(
    const Root& lhs,
    const Root& rhs,
    double relative_tolerance) {
    const double relative_window = relative_tolerance *
        std::max({1.0, std::abs(lhs.density), std::abs(rhs.density)});
    const double residual_window = 2.0 *
        (density_uncertainty(lhs) + density_uncertainty(rhs));
    return std::abs(lhs.density - rhs.density) <=
        std::max(relative_window, residual_window);
}

inline std::vector<Root> merge_roots(
    const std::vector<Root>& first,
    const std::vector<Root>& second,
    double relative_tolerance) {
    std::vector<Root> merged = first;
    for (const auto& candidate : second) {
        auto existing = std::find_if(
            merged.begin(), merged.end(),
            [&](const Root& root) {
                return same_density(
                    root.density, candidate.density, relative_tolerance);
            });
        if (existing == merged.end()) {
            merged.push_back(candidate);
        } else if (std::abs(candidate.residual) <
                   std::abs(existing->residual)) {
            *existing = candidate;
        }
    }
    std::sort(
        merged.begin(), merged.end(),
        [](const Root& lhs, const Root& rhs) {
            return lhs.density < rhs.density;
        });
    return merged;
}

// Stable-set comparison at an explicit dimensionless threshold, applied
// symmetrically to the reference and the returned roots.  Returns the number
// of reference stable roots not matched and returned stable roots not
// matching, so that the exact-stable-set criterion can be evaluated at
// several thresholds from one run.
inline void compare_stable_at_tolerance(
    const std::vector<Root>& reference,
    const std::vector<Root>& returned,
    double relative_tolerance,
    double chi_tolerance,
    int& missed_stable,
    int& extra_stable) {
    missed_stable = 0;
    extra_stable = 0;
    std::vector<bool> reference_matched(reference.size(), false);
    std::vector<bool> returned_matched(returned.size(), false);
    for (std::size_t r = 0; r < returned.size(); ++r) {
        std::size_t best = reference.size();
        double best_difference = std::numeric_limits<double>::infinity();
        for (std::size_t i = 0; i < reference.size(); ++i) {
            if (reference_matched[i] ||
                !same_root(reference[i], returned[r], relative_tolerance)) {
                continue;
            }
            const double difference =
                std::abs(reference[i].density - returned[r].density);
            if (difference < best_difference) {
                best = i;
                best_difference = difference;
            }
        }
        if (best == reference.size()) {
            continue;
        }
        reference_matched[best] = true;
        returned_matched[r] = true;
    }
    for (std::size_t i = 0; i < reference.size(); ++i) {
        if (!reference_matched[i] &&
            classify(reference[i], chi_tolerance) == Stability::stable) {
            ++missed_stable;
        }
    }
    for (std::size_t r = 0; r < returned.size(); ++r) {
        if (!returned_matched[r] &&
            classify(returned[r], chi_tolerance) == Stability::stable) {
            ++extra_stable;
        }
    }
}

inline RootComparison compare_roots(
    const std::vector<Root>& reference,
    const std::vector<Root>& returned,
    double relative_tolerance) {
    RootComparison result;
    std::vector<bool> reference_matched(reference.size(), false);
    std::vector<bool> returned_matched(returned.size(), false);

    for (std::size_t returned_index = 0;
         returned_index < returned.size();
         ++returned_index) {
        std::size_t best = reference.size();
        double best_difference = std::numeric_limits<double>::infinity();
        for (std::size_t reference_index = 0;
             reference_index < reference.size();
             ++reference_index) {
            if (reference_matched[reference_index] ||
                !same_root(
                    reference[reference_index],
                    returned[returned_index],
                    relative_tolerance)) {
                continue;
            }
            const double difference = std::abs(
                reference[reference_index].density -
                returned[returned_index].density);
            if (difference < best_difference) {
                best = reference_index;
                best_difference = difference;
            }
        }
        if (best == reference.size()) {
            continue;
        }
        reference_matched[best] = true;
        returned_matched[returned_index] = true;
        ++result.matched;
        if (reference[best].tangent) {
            ++result.matched_tangent;
        }
    }

    for (std::size_t index = 0; index < reference.size(); ++index) {
        if (reference_matched[index]) {
            continue;
        }
        ++result.missed;
        if (reference[index].mechanically_stable) {
            ++result.missed_stable;
        }
        if (reference[index].tangent) {
            ++result.missed_tangent;
        }
    }
    for (std::size_t index = 0; index < returned.size(); ++index) {
        if (returned_matched[index]) {
            continue;
        }
        ++result.extra;
        if (returned[index].mechanically_stable) {
            ++result.extra_stable;
        }
    }

    result.complete_all = result.missed == 0 && result.extra == 0;
    result.complete_stable = result.missed_stable == 0 &&
        result.extra_stable == 0;
    result.complete_tangent = result.missed_tangent == 0;
    return result;
}

}  // namespace solver_benchmark
