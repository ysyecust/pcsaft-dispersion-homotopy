#pragma once

#include "homotopy_solver.hpp"

#include <string>
#include <vector>

namespace benchmark_states {

struct StateDefinition {
    std::string state_id;
    std::string dataset;
    std::string system;
    int component_count = 0;
    std::string composition;
    std::vector<double> mole_fractions;
    double temperature = 0.0;
    double pressure = 0.0;
    homotopy::EoSHomotopyInterface eos;
};

std::vector<StateDefinition> build_all_path_states();
std::vector<StateDefinition> build_scaleup_isotherm_seeds();

}  // namespace benchmark_states
