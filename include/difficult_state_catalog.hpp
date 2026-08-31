#pragma once

#include "benchmark_state_catalog.hpp"
#include "stationary_root_solver.hpp"

#include <limits>
#include <string>
#include <vector>

namespace difficult_states {

enum class TargetClass {
    interior_three_root,
    lower_tangent,
    upper_tangent,
};

inline const char* target_class_name(TargetClass value) {
    switch (value) {
        case TargetClass::interior_three_root:
            return "interior_three_root";
        case TargetClass::lower_tangent:
            return "lower_tangent";
        case TargetClass::upper_tangent:
            return "upper_tangent";
    }
    return "unknown";
}

struct TargetConfig {
    std::vector<double> interior_positions = {
        1e-5, 1e-4, 1e-3, 0.01, 0.05, 0.20, 0.50,
        0.80, 0.95, 0.99, 0.999, 0.9999, 0.99999,
    };
    bool include_tangent_targets = false;
    int stationary_linear_intervals = 2048;
    int stationary_log_intervals = 1024;
    double derivative_tolerance = 1e-9;
    double pressure_tolerance = 1e-6;
    double density_tolerance = 1e-11;
    double minimum_pressure_span = 1e-4;
};

struct GeneratedState {
    benchmark_states::StateDefinition state;
    std::string source_isotherm_id;
    TargetClass target_class = TargetClass::interior_three_root;
    int stationary_pair_index = -1;
    double target_position = std::numeric_limits<double>::quiet_NaN();
    double stationary_density_low = std::numeric_limits<double>::quiet_NaN();
    double stationary_density_high = std::numeric_limits<double>::quiet_NaN();
    double stationary_pressure_low = std::numeric_limits<double>::quiet_NaN();
    double stationary_pressure_high = std::numeric_limits<double>::quiet_NaN();
};

struct TargetGenerationResult {
    bool resolved = false;
    std::string reason;
    std::vector<stationary_roots::StationaryPoint> stationary_points;
    std::vector<GeneratedState> states;
    int pressure_evaluations = 0;
    int derivative_evaluations = 0;
};

struct CatalogConfig {
    std::size_t pure_states = 5000;
    std::size_t binary_states = 15000;
    std::size_t multicomponent_states = 15000;
    std::size_t process18_states = 15000;
    double tangent_fraction = 0.0;
    TargetConfig target_config;
};

struct CatalogResult {
    bool complete = false;
    std::string reason;
    std::vector<GeneratedState> states;
    std::size_t interior_states = 0;
    std::size_t tangent_states = 0;
    std::size_t scanned_isotherms = 0;
    std::size_t eligible_isotherms = 0;
};

TargetGenerationResult generate_targets(
    const benchmark_states::StateDefinition& isotherm,
    const TargetConfig& config = {});

CatalogResult build_catalog_from_seeds(
    const std::vector<benchmark_states::StateDefinition>& seeds,
    const CatalogConfig& config = {});

CatalogResult build_catalog(const CatalogConfig& config = {});

}  // namespace difficult_states
