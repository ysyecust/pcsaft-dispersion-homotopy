#include "difficult_state_catalog.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <map>
#include <sstream>

namespace difficult_states {
namespace {

std::string generated_id(
    const std::string& base,
    int pair_index,
    const std::string& suffix,
    int ordinal) {
    std::ostringstream output;
    output << base << "_sp" << std::setw(2) << std::setfill('0') << pair_index
           << '_' << suffix << std::setw(4) << std::setfill('0') << ordinal;
    return output.str();
}

double derivative_scale(
    const benchmark_states::StateDefinition& state) {
    const double lower = state.eos.rho_min;
    const double upper = state.eos.rho_max * (1.0 - 1e-10);
    const double middle = 0.5 * (lower + upper);
    return std::max({
        1.0,
        std::abs(state.eos.dPdrho(lower, 1.0)),
        std::abs(state.eos.dPdrho(middle, 1.0)),
        std::abs(state.eos.dPdrho(upper, 1.0)),
    });
}

stationary_roots::StationaryPoint refine_stationary_point(
    const benchmark_states::StateDefinition& state,
    stationary_roots::StationaryPoint point,
    double scale,
    int& derivative_evaluations,
    int& pressure_evaluations) {
    if (!point.isolated_by_sign_change ||
        !std::isfinite(point.bracket_lower) ||
        !std::isfinite(point.bracket_upper) ||
        !(point.bracket_upper > point.bracket_lower)) {
        return point;
    }

    double lower = point.bracket_lower;
    double upper = point.bracket_upper;
    double derivative_lower = state.eos.dPdrho(lower, 1.0);
    double derivative_upper = state.eos.dPdrho(upper, 1.0);
    derivative_evaluations += 2;
    if (!std::isfinite(derivative_lower) ||
        !std::isfinite(derivative_upper) ||
        derivative_lower * derivative_upper > 0.0) {
        return point;
    }

    const double derivative_tolerance = std::max(
        1e-12,
        1e-14 * std::max(1.0, std::abs(scale)));
    for (int iteration = 0; iteration < 200; ++iteration) {
        const double middle = 0.5 * (lower + upper);
        if (middle == lower || middle == upper) {
            break;
        }
        const double derivative_middle = state.eos.dPdrho(middle, 1.0);
        ++derivative_evaluations;
        if (!std::isfinite(derivative_middle)) {
            return point;
        }
        point.density = middle;
        point.derivative = derivative_middle;
        if (std::abs(derivative_middle) <= derivative_tolerance) {
            lower = middle;
            upper = middle;
            break;
        }
        if ((derivative_lower <= 0.0 && derivative_middle >= 0.0) ||
            (derivative_lower >= 0.0 && derivative_middle <= 0.0)) {
            upper = middle;
            derivative_upper = derivative_middle;
        } else {
            lower = middle;
            derivative_lower = derivative_middle;
        }
    }
    point.bracket_lower = lower;
    point.bracket_upper = upper;
    point.pressure_residual =
        state.eos.P(point.density, 1.0) - state.pressure;
    ++pressure_evaluations;
    return point;
}

}  // namespace

TargetGenerationResult generate_targets(
    const benchmark_states::StateDefinition& isotherm,
    const TargetConfig& config) {
    TargetGenerationResult output;
    if (!isotherm.eos.P || !isotherm.eos.dPdrho ||
        !(isotherm.eos.rho_max > isotherm.eos.rho_min) ||
        config.stationary_linear_intervals < 1 ||
        config.stationary_log_intervals < 0) {
        output.reason = "invalid_isotherm_or_configuration";
        return output;
    }
    for (double position : config.interior_positions) {
        if (!std::isfinite(position) || !(position > 0.0 && position < 1.0)) {
            output.reason = "interior_position_out_of_range";
            return output;
        }
    }

    stationary_roots::Equation equation;
    equation.pressure = [&](double density) {
        return isotherm.eos.P(density, 1.0);
    };
    equation.derivative = [&](double density) {
        return isotherm.eos.dPdrho(density, 1.0);
    };
    equation.target_pressure = isotherm.pressure;
    equation.density_min = isotherm.eos.rho_min;
    equation.density_max = isotherm.eos.rho_max * (1.0 - 1e-10);
    equation.derivative_scale = derivative_scale(isotherm);

    stationary_roots::SolverConfig solver_config;
    solver_config.linear_intervals = config.stationary_linear_intervals;
    solver_config.log_intervals = config.stationary_log_intervals;
    solver_config.derivative_tolerance = config.derivative_tolerance;
    solver_config.pressure_tolerance = config.pressure_tolerance;
    solver_config.density_tolerance = config.density_tolerance;
    solver_config.relative_merge_tolerance = 1e-10;
    const auto stationary = stationary_roots::solve(equation, solver_config);
    output.pressure_evaluations = stationary.pressure_evaluations;
    output.derivative_evaluations = stationary.derivative_evaluations;
    output.stationary_points = stationary.stationary_points;
    for (auto& point : output.stationary_points) {
        point = refine_stationary_point(
            isotherm,
            point,
            equation.derivative_scale,
            output.derivative_evaluations,
            output.pressure_evaluations);
    }
    if (!stationary.success || stationary.invalid_evaluation) {
        output.reason = stationary.invalid_evaluation
            ? "nonfinite_stationary_search"
            : "stationary_search_failed";
        return output;
    }
    if (output.stationary_points.size() < 2) {
        output.reason = "fewer_than_two_stationary_points";
        output.resolved = true;
        return output;
    }

    int pair_index = 0;
    for (std::size_t index = 1;
         index < output.stationary_points.size();
         ++index) {
        const auto& first = output.stationary_points[index - 1];
        const auto& second = output.stationary_points[index];
        const double first_pressure = equation.pressure(first.density);
        const double second_pressure = equation.pressure(second.density);
        output.pressure_evaluations += 2;
        if (!std::isfinite(first_pressure) || !std::isfinite(second_pressure)) {
            output.reason = "nonfinite_stationary_pressure";
            return output;
        }
        const double pressure_low = std::min(first_pressure, second_pressure);
        const double pressure_high = std::max(first_pressure, second_pressure);
        const double boundary_pressure = equation.pressure(equation.density_min);
        ++output.pressure_evaluations;
        if (!std::isfinite(boundary_pressure)) {
            output.reason = "nonfinite_low_density_pressure";
            return output;
        }
        const double feasible_pressure_low = std::max({
            pressure_low,
            boundary_pressure,
            1.0,
        });
        if (pressure_high - feasible_pressure_low <=
            config.minimum_pressure_span) {
            continue;
        }

        int ordinal = 0;
        for (double position : config.interior_positions) {
            GeneratedState row;
            row.state = isotherm;
            row.source_isotherm_id = isotherm.state_id;
            row.state.state_id = generated_id(
                isotherm.state_id, pair_index, "p", ordinal);
            row.state.dataset = isotherm.dataset + "_multiple_root";
            row.state.pressure = feasible_pressure_low +
                position * (pressure_high - feasible_pressure_low);
            row.target_class = TargetClass::interior_three_root;
            row.stationary_pair_index = pair_index;
            row.target_position = position;
            row.stationary_density_low = first.density;
            row.stationary_density_high = second.density;
            row.stationary_pressure_low = pressure_low;
            row.stationary_pressure_high = pressure_high;
            output.states.push_back(std::move(row));
            ++ordinal;
        }

        if (config.include_tangent_targets) {
            if (pressure_low > 0.0 && pressure_low >= boundary_pressure) {
                GeneratedState lower;
                lower.state = isotherm;
                lower.source_isotherm_id = isotherm.state_id;
                lower.state.state_id = generated_id(
                    isotherm.state_id, pair_index, "tl", 0);
                lower.state.dataset = isotherm.dataset + "_tangent";
                lower.state.pressure = pressure_low;
                lower.target_class = TargetClass::lower_tangent;
                lower.stationary_pair_index = pair_index;
                lower.target_position = 0.0;
                lower.stationary_density_low = first.density;
                lower.stationary_density_high = second.density;
                lower.stationary_pressure_low = pressure_low;
                lower.stationary_pressure_high = pressure_high;
                output.states.push_back(std::move(lower));
            }

            GeneratedState upper;
            upper.state = isotherm;
            upper.source_isotherm_id = isotherm.state_id;
            upper.state.state_id = generated_id(
                isotherm.state_id, pair_index, "tu", 0);
            upper.state.dataset = isotherm.dataset + "_tangent";
            upper.state.pressure = pressure_high;
            upper.target_class = TargetClass::upper_tangent;
            upper.stationary_pair_index = pair_index;
            upper.target_position = 1.0;
            upper.stationary_density_low = first.density;
            upper.stationary_density_high = second.density;
            upper.stationary_pressure_low = pressure_low;
            upper.stationary_pressure_high = pressure_high;
            output.states.push_back(std::move(upper));
        }
        ++pair_index;
    }

    output.resolved = true;
    output.reason = output.states.empty()
        ? "no_stationary_pressure_interval"
        : "generated";
    return output;
}

namespace {

enum class Group {
    pure,
    binary,
    multicomponent,
    process18,
    unknown,
};

Group group_of(const std::string& dataset) {
    if (dataset == "scaleup_pure") {
        return Group::pure;
    }
    if (dataset == "scaleup_binary") {
        return Group::binary;
    }
    if (dataset == "scaleup_multicomponent") {
        return Group::multicomponent;
    }
    if (dataset == "scaleup_process18") {
        return Group::process18;
    }
    return Group::unknown;
}

std::size_t requested_for(Group group, const CatalogConfig& config) {
    switch (group) {
        case Group::pure:
            return config.pure_states;
        case Group::binary:
            return config.binary_states;
        case Group::multicomponent:
            return config.multicomponent_states;
        case Group::process18:
            return config.process18_states;
        case Group::unknown:
            return 0;
    }
    return 0;
}

std::uint64_t fnv1a(const std::string& value) {
    std::uint64_t hash = 1469598103934665603ULL;
    for (unsigned char character : value) {
        hash ^= character;
        hash *= 1099511628211ULL;
    }
    return hash;
}

struct Reservoir {
    std::size_t capacity = 0;
    std::size_t seen = 0;
    std::vector<GeneratedState> rows;

    void consider(GeneratedState row) {
        ++seen;
        if (capacity == 0) {
            return;
        }
        if (rows.size() < capacity) {
            rows.push_back(std::move(row));
            return;
        }
        const std::size_t candidate = static_cast<std::size_t>(
            fnv1a(row.state.state_id) % seen);
        if (candidate < capacity) {
            rows[candidate] = std::move(row);
        }
    }
};

struct GroupReservoirs {
    Reservoir interior;
    Reservoir tangent;
};

}  // namespace

CatalogResult build_catalog_from_seeds(
    const std::vector<benchmark_states::StateDefinition>& seeds,
    const CatalogConfig& config) {
    CatalogResult result;
    if (!(config.tangent_fraction >= 0.0 && config.tangent_fraction <= 1.0)) {
        result.reason = "tangent_fraction_out_of_range";
        return result;
    }

    std::map<Group, GroupReservoirs> reservoirs;
    for (Group group : {
             Group::pure,
             Group::binary,
             Group::multicomponent,
             Group::process18}) {
        const std::size_t requested = requested_for(group, config);
        const std::size_t tangent = static_cast<std::size_t>(
            std::llround(requested * config.tangent_fraction));
        reservoirs[group].tangent.capacity = tangent;
        reservoirs[group].interior.capacity = requested - tangent;
    }

    for (const auto& seed : seeds) {
        const Group group = group_of(seed.dataset);
        if (group == Group::unknown || requested_for(group, config) == 0) {
            continue;
        }
        ++result.scanned_isotherms;
        auto generated = generate_targets(seed, config.target_config);
        if (!generated.resolved || generated.states.empty()) {
            continue;
        }
        ++result.eligible_isotherms;
        for (auto& row : generated.states) {
            if (row.target_class == TargetClass::interior_three_root) {
                reservoirs[group].interior.consider(std::move(row));
            } else {
                reservoirs[group].tangent.consider(std::move(row));
            }
        }
    }

    for (Group group : {
             Group::pure,
             Group::binary,
             Group::multicomponent,
             Group::process18}) {
        auto& selected = reservoirs[group];
        if (selected.interior.rows.size() != selected.interior.capacity ||
            selected.tangent.rows.size() != selected.tangent.capacity) {
            result.reason = "insufficient_candidates";
            return result;
        }
        result.interior_states += selected.interior.rows.size();
        result.tangent_states += selected.tangent.rows.size();
        for (auto& row : selected.interior.rows) {
            result.states.push_back(std::move(row));
        }
        for (auto& row : selected.tangent.rows) {
            result.states.push_back(std::move(row));
        }
    }

    std::sort(
        result.states.begin(),
        result.states.end(),
        [](const GeneratedState& lhs, const GeneratedState& rhs) {
            return lhs.state.state_id < rhs.state.state_id;
        });
    const auto duplicate = std::adjacent_find(
        result.states.begin(),
        result.states.end(),
        [](const GeneratedState& lhs, const GeneratedState& rhs) {
            return lhs.state.state_id == rhs.state.state_id;
        });
    if (duplicate != result.states.end()) {
        result.reason = "duplicate_state_identifier";
        result.states.clear();
        return result;
    }

    result.complete = true;
    result.reason = "complete";
    return result;
}

CatalogResult build_catalog(const CatalogConfig& config) {
    return build_catalog_from_seeds(
        benchmark_states::build_scaleup_isotherm_seeds(), config);
}

}  // namespace difficult_states
