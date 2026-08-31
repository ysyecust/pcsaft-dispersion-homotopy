#pragma once

#include "difficult_state_catalog.hpp"

#include <cstddef>
#include <string>

namespace all_root_states {

struct Validation {
    bool valid = false;
    std::string reason;
    std::size_t pure_states = 0;
    std::size_t binary_states = 0;
    std::size_t multicomponent_states = 0;
    std::size_t process18_states = 0;
    std::size_t multi_root_target_states = 0;
    std::size_t tangent_target_states = 0;
};

difficult_states::CatalogConfig formal_config();
difficult_states::CatalogConfig validation_config();
difficult_states::CatalogConfig quick_config();
difficult_states::CatalogResult build(
    const difficult_states::CatalogConfig& config);
std::string group_name(
    const benchmark_states::StateDefinition& state);
Validation validate(
    const difficult_states::CatalogResult& catalog,
    const difficult_states::CatalogConfig& config);

}  // namespace all_root_states
