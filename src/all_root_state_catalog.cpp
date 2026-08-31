#include "all_root_state_catalog.hpp"

#include <set>

namespace all_root_states {

difficult_states::CatalogConfig formal_config() {
    difficult_states::CatalogConfig config;
    config.pure_states = 10000;
    config.binary_states = 30000;
    config.multicomponent_states = 30000;
    config.process18_states = 30000;
    config.tangent_fraction = 0.02;
    config.target_config.include_tangent_targets = true;
    config.target_config.stationary_linear_intervals = 2048;
    config.target_config.stationary_log_intervals = 1024;
    return config;
}

difficult_states::CatalogConfig validation_config() {
    auto config = formal_config();
    config.target_config.interior_positions = {
        3e-5, 3e-4, 3e-3, 0.03, 0.10, 0.30, 0.70,
        0.90, 0.97, 0.997, 0.9997, 0.99997, 0.999997,
    };
    return config;
}

difficult_states::CatalogConfig quick_config() {
    auto config = formal_config();
    config.pure_states = 4;
    config.binary_states = 12;
    config.multicomponent_states = 12;
    config.process18_states = 12;
    config.tangent_fraction = 0.10;
    config.target_config.stationary_linear_intervals = 256;
    config.target_config.stationary_log_intervals = 128;
    return config;
}

difficult_states::CatalogResult build(
    const difficult_states::CatalogConfig& config) {
    return difficult_states::build_catalog(config);
}

std::string group_name(
    const benchmark_states::StateDefinition& state) {
    if (state.component_count == 1) {
        return "pure";
    }
    if (state.component_count == 2) {
        return "binary";
    }
    if (state.component_count >= 3 && state.component_count <= 4) {
        return "three_to_four_component";
    }
    if (state.component_count == 18) {
        return "eighteen_component";
    }
    return "outside_design";
}

Validation validate(
    const difficult_states::CatalogResult& catalog,
    const difficult_states::CatalogConfig& config) {
    Validation result;
    if (!catalog.complete) {
        result.reason = "catalog_generation_incomplete";
        return result;
    }

    std::set<std::string> identifiers;
    for (const auto& row : catalog.states) {
        if (!identifiers.insert(row.state.state_id).second) {
            result.reason = "duplicate_state_identifier";
            return result;
        }
        const auto group = group_name(row.state);
        if (group == "pure") {
            ++result.pure_states;
        } else if (group == "binary") {
            ++result.binary_states;
        } else if (group == "three_to_four_component") {
            ++result.multicomponent_states;
        } else if (group == "eighteen_component") {
            ++result.process18_states;
        } else {
            result.reason = "state_outside_frozen_groups";
            return result;
        }

        if (row.target_class ==
            difficult_states::TargetClass::interior_three_root) {
            ++result.multi_root_target_states;
        } else {
            ++result.tangent_target_states;
        }
    }

    result.valid = result.pure_states == config.pure_states &&
        result.binary_states == config.binary_states &&
        result.multicomponent_states == config.multicomponent_states &&
        result.process18_states == config.process18_states &&
        result.multi_root_target_states == catalog.interior_states &&
        result.tangent_target_states == catalog.tangent_states;
    result.reason = result.valid ? "valid" : "quota_mismatch";
    return result;
}

}  // namespace all_root_states
