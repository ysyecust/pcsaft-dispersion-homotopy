#include "benchmark_state_catalog.hpp"

#include "pcsaft_accurate.hpp"
#include "pcsaft_mixture.hpp"

#include <iomanip>
#include <array>
#include <cmath>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace benchmark_states {
namespace {

using homotopy::EoSHomotopyInterface;

std::string composition_string(const std::vector<double>& composition) {
    std::ostringstream output;
    output << std::setprecision(9);
    for (std::size_t index = 0; index < composition.size(); ++index) {
        if (index > 0) {
            output << ';';
        }
        output << composition[index];
    }
    return output.str();
}

std::string make_state_id(const std::string& dataset, std::size_t ordinal) {
    std::ostringstream output;
    output << dataset << '_' << std::setw(4) << std::setfill('0') << ordinal;
    return output.str();
}

EoSHomotopyInterface make_pure_eos(
    const pcsaft_acc::Component& component,
    double temperature) {
    const double rho_max = pcsaft_acc::max_density(component, temperature);
    EoSHomotopyInterface eos;
    eos.T = temperature;
    eos.rho_min = 1e-3;
    eos.rho_max = rho_max;
    eos.rho_ref = rho_max;
    eos.P = [component, temperature](double rho, double lambda) {
        if (rho <= 0.0) {
            return 0.0;
        }
        return pcsaft_acc::evaluate(component, rho, temperature, lambda).P;
    };
    eos.dPdrho = [component, temperature](double rho, double lambda) {
        if (rho <= 0.0) {
            return 1e10;
        }
        return pcsaft_acc::evaluate(component, rho, temperature, lambda).dPdrho;
    };
    eos.dPdlam = [component, temperature](double rho, double) {
        if (rho <= 0.0) {
            return 0.0;
        }
        return pcsaft_acc::evaluate(component, rho, temperature, 1.0).P -
               pcsaft_acc::evaluate(component, rho, temperature, 0.0).P;
    };
    eos.evaluate_all = [component, temperature](double rho, double lambda) {
        if (rho <= 0.0) {
            return homotopy::EoSEvaluation{0.0, 1e10, 0.0};
        }
        const auto zero =
            pcsaft_acc::evaluate(component, rho, temperature, 0.0);
        const auto one =
            pcsaft_acc::evaluate(component, rho, temperature, 1.0);
        return homotopy::EoSEvaluation{
            zero.P + lambda * (one.P - zero.P),
            zero.dPdrho + lambda * (one.dPdrho - zero.dPdrho),
            one.P - zero.P,
        };
    };
    return eos;
}

EoSHomotopyInterface make_mixture_eos(
    const pcsaft_mix::MixtureParams& parameters,
    const std::vector<double>& composition,
    double temperature) {
    const double rho_max =
        pcsaft_mix::max_density_mix(parameters, composition, temperature);
    EoSHomotopyInterface eos;
    eos.T = temperature;
    eos.rho_min = 1e-3;
    eos.rho_max = rho_max;
    eos.rho_ref = rho_max;
    eos.P = [parameters, composition, temperature](double rho, double lambda) {
        if (rho <= 0.0) {
            return 0.0;
        }
        return pcsaft_mix::evaluate_P(
            parameters, composition, rho, temperature, lambda);
    };
    eos.dPdrho =
        [parameters, composition, temperature](double rho, double lambda) {
            if (rho <= 0.0) {
                return 1e10;
            }
            return pcsaft_mix::evaluate_mixture(
                       parameters, composition, rho, temperature, lambda)
                .dPdrho;
        };
    eos.dPdlam =
        [parameters, composition, temperature](double rho, double) {
            if (rho <= 0.0) {
                return 0.0;
            }
            return pcsaft_mix::evaluate_P(
                       parameters, composition, rho, temperature, 1.0) -
                   pcsaft_mix::evaluate_P(
                       parameters, composition, rho, temperature, 0.0);
        };
    eos.evaluate_all =
        [parameters, composition, temperature](double rho, double lambda) {
            if (rho <= 0.0) {
                return homotopy::EoSEvaluation{0.0, 1e10, 0.0};
            }
            const auto current = pcsaft_mix::evaluate_mixture(
                parameters, composition, rho, temperature, lambda);
            const double pressure_zero = lambda == 0.0
                ? current.P
                : pcsaft_mix::evaluate_P(
                      parameters, composition, rho, temperature, 0.0);
            const double pressure_one = lambda == 1.0
                ? current.P
                : pcsaft_mix::evaluate_P(
                      parameters, composition, rho, temperature, 1.0);
            return homotopy::EoSEvaluation{
                current.P,
                current.dPdrho,
                pressure_one - pressure_zero,
            };
        };
    return eos;
}

void append_state(
    std::vector<StateDefinition>& states,
    const std::string& dataset,
    const std::string& system,
    const std::vector<double>& composition,
    double temperature,
    double pressure,
    EoSHomotopyInterface eos) {
    states.push_back({
        make_state_id(dataset, states.size() + 1),
        dataset,
        system,
        static_cast<int>(composition.size()),
        composition_string(composition),
        composition,
        temperature,
        pressure,
        std::move(eos),
    });
}

void add_grid(
    std::vector<StateDefinition>& states,
    const std::string& dataset,
    const std::string& system,
    const pcsaft_mix::MixtureParams& parameters,
    std::vector<double> composition,
    double temperature_lo,
    double temperature_hi,
    int n_temperature,
    double pressure_lo,
    double pressure_hi,
    int n_pressure) {
    const double sum = std::accumulate(
        composition.begin(), composition.end(), 0.0);
    for (double& value : composition) {
        value /= sum;
    }

    for (int temperature_index = 0;
         temperature_index < n_temperature;
         ++temperature_index) {
        const double temperature = n_temperature == 1
            ? temperature_lo
            : temperature_lo + (temperature_hi - temperature_lo) *
                  temperature_index / (n_temperature - 1.0);
        for (int pressure_index = 0;
             pressure_index < n_pressure;
             ++pressure_index) {
            const double pressure = n_pressure == 1
                ? pressure_lo
                : pressure_lo + (pressure_hi - pressure_lo) *
                      pressure_index / (n_pressure - 1.0);
            append_state(
                states,
                dataset,
                system,
                composition,
                temperature,
                pressure,
                make_mixture_eos(parameters, composition, temperature));
        }
    }
}

pcsaft_mix::MixtureParams process_18_parameters() {
    pcsaft_mix::MixtureParams parameters;
    parameters.nc = 18;
    parameters.m = {
        1.0000, 1.6069, 2.0020, 2.2616, 2.3316, 2.6896,
        2.5620, 3.0576, 3.4831, 3.8176, 2.5303, 1.5930,
        1.9597, 2.2864, 2.4653, 2.8149, 1.2053, 2.5692,
    };
    parameters.sigma = {
        3.7039, 3.5206, 3.6184, 3.7574, 3.7086, 3.7729,
        3.8296, 3.7983, 3.8049, 3.8373, 3.8499, 3.4450,
        3.5356, 3.6431, 3.6478, 3.7169, 3.3130, 2.5637,
    };
    parameters.eps_k = {
        150.03, 191.42, 208.11, 216.53, 222.88, 231.20,
        230.75, 236.77, 238.40, 242.78, 278.11, 176.47,
        207.19, 222.00, 287.35, 285.69, 90.96, 152.10,
    };
    parameters.kij.assign(18, std::vector<double>(18, 0.0));
    auto set = [&](int lhs, int rhs, double value) {
        parameters.kij[lhs][rhs] = value;
        parameters.kij[rhs][lhs] = value;
    };

    set(0, 1, 0.000422); set(0, 2, 0.024151); set(0, 3, 0.046072);
    set(0, 4, 0.022644); set(0, 5, 0.015811); set(0, 7, 0.022073);
    set(0, 8, -0.006513); set(0, 9, 0.095823); set(0, 11, 0.017405);
    set(0, 12, 0.028900); set(0, 14, 0.031339); set(0, 15, 0.081803);
    set(0, 17, 0.093300);
    set(1, 2, 0.001695); set(1, 3, 0.005512); set(1, 4, 0.005322);
    set(1, 5, 0.014253); set(1, 7, -0.043356); set(1, 8, 0.018005);
    set(1, 9, -0.154972); set(1, 11, 0.003211); set(1, 12, 0.011799);
    set(1, 14, 0.036223); set(1, 15, 0.022626); set(1, 17, 0.136300);
    set(2, 3, -0.002909); set(2, 4, -0.002076); set(2, 5, -0.004328);
    set(2, 7, 0.003052); set(2, 8, 0.036109); set(2, 12, 0.003000);
    set(2, 14, 0.019558); set(2, 15, 0.033030); set(2, 17, 0.128900);
    set(3, 4, -0.003775); set(3, 12, -0.021258);
    set(4, 5, 0.038188); set(4, 7, 0.012447); set(4, 8, -0.006900);
    set(4, 11, 0.144538); set(4, 13, 0.002229); set(4, 14, 0.009900);
    set(4, 17, 0.143000);
    set(5, 8, 0.015871); set(5, 9, -0.002286); set(5, 14, 0.022415);
    set(5, 17, 0.131100);
    set(7, 8, 0.002107); set(7, 14, 0.014592); set(7, 17, 0.117800);
    set(8, 11, 0.016450); set(8, 14, 0.007088); set(8, 17, 0.110000);
    set(9, 14, -0.002854);
    set(11, 14, 0.028355); set(11, 17, 0.053300);
    set(12, 13, 0.003643); set(12, 16, 0.108200); set(12, 17, 0.083110);
    set(13, 17, 0.059600);
    set(14, 15, 0.020000); set(14, 17, 0.088060);
    set(15, 16, 0.199600); set(15, 17, 0.113000);
    return parameters;
}

}  // namespace

std::vector<StateDefinition> build_all_path_states() {
    std::vector<StateDefinition> states;
    states.reserve(2180);

    const double critical_temperatures[] = {
        190.56, 305.32, 369.83, 425.12, 304.21,
    };
    const double pressures[] = {5e5, 10e5, 20e5, 40e5};
    for (std::size_t component_index = 0;
         component_index < pcsaft_acc::COMPONENTS.size();
         ++component_index) {
        const auto component = pcsaft_acc::COMPONENTS[component_index];
        for (int temperature_index = 0;
             temperature_index <= 30;
             ++temperature_index) {
            const double temperature =
                0.5 * critical_temperatures[component_index] +
                (1.3 - 0.5) * critical_temperatures[component_index] *
                    temperature_index / 30.0;
            for (double pressure : pressures) {
                append_state(
                    states,
                    "pure_diagnostic",
                    component.name,
                    {1.0},
                    temperature,
                    pressure,
                    make_pure_eos(component, temperature));
            }
        }
    }

    pcsaft_mix::MixtureParams methane_co2;
    methane_co2.nc = 2;
    methane_co2.m = {1.0000, 2.0729};
    methane_co2.sigma = {3.7039, 2.7852};
    methane_co2.eps_k = {150.03, 169.21};
    methane_co2.kij = {{0.0, 0.05}, {0.05, 0.0}};

    pcsaft_mix::MixtureParams methane_decane;
    methane_decane.nc = 2;
    methane_decane.m = {1.0000, 4.6627};
    methane_decane.sigma = {3.7039, 3.8384};
    methane_decane.eps_k = {150.03, 243.87};
    methane_decane.kij = {{0.0, 0.04}, {0.04, 0.0}};

    pcsaft_mix::MixtureParams co2_propane;
    co2_propane.nc = 2;
    co2_propane.m = {2.0729, 2.0020};
    co2_propane.sigma = {2.7852, 3.6184};
    co2_propane.eps_k = {169.21, 208.11};
    co2_propane.kij = {{0.0, 0.10}, {0.10, 0.0}};

    add_grid(states, "binary_mixture", "CH4/CO2", methane_co2,
             {0.7, 0.3}, 120.0, 250.0, 20, 5e5, 80e5, 10);
    add_grid(states, "binary_mixture", "CH4/n-C10", methane_decane,
             {0.9, 0.1}, 150.0, 400.0, 20, 5e5, 100e5, 10);
    add_grid(states, "binary_mixture", "CO2/propane", co2_propane,
             {0.5, 0.5}, 200.0, 370.0, 20, 5e5, 60e5, 10);

    pcsaft_mix::MixtureParams natural_gas;
    natural_gas.nc = 4;
    natural_gas.m = {1.0000, 1.6069, 2.0020, 2.0729};
    natural_gas.sigma = {3.7039, 3.5206, 3.6184, 2.7852};
    natural_gas.eps_k = {150.03, 191.42, 208.11, 169.21};
    natural_gas.kij = {
        {0.000, 0.003, 0.012, 0.050},
        {0.003, 0.000, 0.001, 0.130},
        {0.012, 0.001, 0.000, 0.135},
        {0.050, 0.130, 0.135, 0.000},
    };

    pcsaft_mix::MixtureParams co2_rich;
    co2_rich.nc = 3;
    co2_rich.m = {2.0729, 1.0000, 1.6069};
    co2_rich.sigma = {2.7852, 3.7039, 3.5206};
    co2_rich.eps_k = {169.21, 150.03, 191.42};
    co2_rich.kij = {
        {0.000, 0.050, 0.130},
        {0.050, 0.000, 0.003},
        {0.130, 0.003, 0.000},
    };

    pcsaft_mix::MixtureParams heavy_hydrocarbon;
    heavy_hydrocarbon.nc = 3;
    heavy_hydrocarbon.m = {1.0000, 2.0020, 4.6627};
    heavy_hydrocarbon.sigma = {3.7039, 3.6184, 3.8384};
    heavy_hydrocarbon.eps_k = {150.03, 208.11, 243.87};
    heavy_hydrocarbon.kij = {
        {0.000, 0.012, 0.040},
        {0.012, 0.000, 0.008},
        {0.040, 0.008, 0.000},
    };

    add_grid(states, "multicomponent_mixture", "Natural gas (4-comp)",
             natural_gas, {0.70, 0.15, 0.05, 0.10},
             150.0, 350.0, 20, 10e5, 80e5, 10);
    add_grid(states, "multicomponent_mixture", "CO2-rich (3-comp)",
             co2_rich, {0.60, 0.25, 0.15},
             180.0, 320.0, 20, 10e5, 80e5, 10);
    add_grid(states, "multicomponent_mixture", "Heavy HC (3-comp)",
             heavy_hydrocarbon, {0.80, 0.10, 0.10},
             200.0, 500.0, 20, 10e5, 100e5, 10);

    const auto process_parameters = process_18_parameters();
    add_grid(states, "process_18_component",
             "Cracked gas (18-comp, light-dominated)", process_parameters,
             {0.15, 0.08, 0.02, 0.005, 0.01, 0.005, 0.003, 0.002,
              0.001, 0.001, 0.001, 0.30, 0.05, 0.02, 0.005, 0.002,
              0.02, 0.01},
             150.0, 400.0, 15, 5e5, 40e5, 8);
    add_grid(states, "process_18_component", "C3 splitter feed (C3-focused)",
             process_parameters,
             {0.005, 0.01, 0.35, 0.005, 0.005, 0.002, 0.001, 0.001,
              0.0, 0.0, 0.0, 0.02, 0.55, 0.01, 0.0, 0.0, 0.005, 0.01},
             230.0, 370.0, 15, 10e5, 25e5, 8);
    add_grid(states, "process_18_component",
             "Deethanizer overhead (C2-rich, high-P)", process_parameters,
             {0.20, 0.12, 0.01, 0.0, 0.005, 0.0, 0.0, 0.0, 0.0,
              0.0, 0.0, 0.45, 0.005, 0.0, 0.0, 0.0, 0.05, 0.03},
             150.0, 320.0, 15, 10e5, 35e5, 8);

    if (states.size() != 2180) {
        throw std::runtime_error(
            "internal state-count error: expected 2180, got " +
            std::to_string(states.size()));
    }
    return states;
}

std::vector<StateDefinition> build_scaleup_isotherm_seeds() {
    std::vector<StateDefinition> states;
    states.reserve(28000);

    const double critical_temperatures[] = {
        190.56, 305.32, 369.83, 425.12, 304.21,
    };
    for (std::size_t component_index = 0;
         component_index < pcsaft_acc::COMPONENTS.size();
         ++component_index) {
        const auto component = pcsaft_acc::COMPONENTS[component_index];
        for (int temperature_index = 0;
             temperature_index < 200;
             ++temperature_index) {
            const double fraction = temperature_index / 199.0;
            const double temperature = critical_temperatures[component_index] *
                (0.55 + (0.9995 - 0.55) * fraction);
            append_state(
                states,
                "scaleup_pure",
                component.name,
                {1.0},
                temperature,
                1e5,
                make_pure_eos(component, temperature));
        }
    }

    auto binary_families = [] {
        struct Family {
            std::string name;
            pcsaft_mix::MixtureParams parameters;
            double temperature_low;
            double temperature_high;
        };
        std::vector<Family> families;

        pcsaft_mix::MixtureParams methane_co2;
        methane_co2.nc = 2;
        methane_co2.m = {1.0000, 2.0729};
        methane_co2.sigma = {3.7039, 2.7852};
        methane_co2.eps_k = {150.03, 169.21};
        methane_co2.kij = {{0.0, 0.05}, {0.05, 0.0}};
        families.push_back({"CH4/CO2", methane_co2, 100.0, 250.0});

        pcsaft_mix::MixtureParams methane_decane;
        methane_decane.nc = 2;
        methane_decane.m = {1.0000, 4.6627};
        methane_decane.sigma = {3.7039, 3.8384};
        methane_decane.eps_k = {150.03, 243.87};
        methane_decane.kij = {{0.0, 0.04}, {0.04, 0.0}};
        families.push_back({"CH4/n-C10", methane_decane, 130.0, 450.0});

        pcsaft_mix::MixtureParams co2_propane;
        co2_propane.nc = 2;
        co2_propane.m = {2.0729, 2.0020};
        co2_propane.sigma = {2.7852, 3.6184};
        co2_propane.eps_k = {169.21, 208.11};
        co2_propane.kij = {{0.0, 0.10}, {0.10, 0.0}};
        families.push_back({"CO2/propane", co2_propane, 180.0, 390.0});
        return families;
    }();

    for (const auto& family : binary_families) {
        for (int composition_index = 0;
             composition_index < 19;
             ++composition_index) {
            const double first = 0.05 + 0.90 * composition_index / 18.0;
            const std::vector<double> composition = {first, 1.0 - first};
            for (int temperature_index = 0;
                 temperature_index < 150;
                 ++temperature_index) {
                const double temperature = family.temperature_low +
                    (family.temperature_high - family.temperature_low) *
                        temperature_index / 149.0;
                append_state(
                    states,
                    "scaleup_binary",
                    family.name,
                    composition,
                    temperature,
                    1e5,
                    make_mixture_eos(
                        family.parameters, composition, temperature));
            }
        }
    }

    auto halton = [](int index, int base) {
        double result = 0.0;
        double factor = 1.0 / base;
        while (index > 0) {
            result += factor * (index % base);
            index /= base;
            factor /= base;
        }
        return result;
    };
    auto simplex_composition = [&](int size, int variant) {
        constexpr std::array<int, 8> primes = {2, 3, 5, 7, 11, 13, 17, 19};
        std::vector<double> composition(static_cast<std::size_t>(size));
        for (int index = 0; index < size; ++index) {
            const double uniform = std::clamp(
                halton(variant + 1, primes[static_cast<std::size_t>(index)]),
                1e-6,
                1.0 - 1e-6);
            composition[static_cast<std::size_t>(index)] =
                0.02 - std::log(uniform);
        }
        const double sum = std::accumulate(
            composition.begin(), composition.end(), 0.0);
        for (double& value : composition) {
            value /= sum;
        }
        return composition;
    };

    struct MultiFamily {
        std::string name;
        pcsaft_mix::MixtureParams parameters;
        double temperature_low;
        double temperature_high;
    };
    std::vector<MultiFamily> multicomponent_families;

    pcsaft_mix::MixtureParams natural_gas;
    natural_gas.nc = 4;
    natural_gas.m = {1.0000, 1.6069, 2.0020, 2.0729};
    natural_gas.sigma = {3.7039, 3.5206, 3.6184, 2.7852};
    natural_gas.eps_k = {150.03, 191.42, 208.11, 169.21};
    natural_gas.kij = {
        {0.000, 0.003, 0.012, 0.050},
        {0.003, 0.000, 0.001, 0.130},
        {0.012, 0.001, 0.000, 0.135},
        {0.050, 0.130, 0.135, 0.000},
    };
    multicomponent_families.push_back(
        {"Natural gas (4-comp)", natural_gas, 120.0, 360.0});

    pcsaft_mix::MixtureParams co2_rich;
    co2_rich.nc = 3;
    co2_rich.m = {2.0729, 1.0000, 1.6069};
    co2_rich.sigma = {2.7852, 3.7039, 3.5206};
    co2_rich.eps_k = {169.21, 150.03, 191.42};
    co2_rich.kij = {
        {0.000, 0.050, 0.130},
        {0.050, 0.000, 0.003},
        {0.130, 0.003, 0.000},
    };
    multicomponent_families.push_back(
        {"CO2-rich (3-comp)", co2_rich, 140.0, 340.0});

    pcsaft_mix::MixtureParams heavy_hydrocarbon;
    heavy_hydrocarbon.nc = 3;
    heavy_hydrocarbon.m = {1.0000, 2.0020, 4.6627};
    heavy_hydrocarbon.sigma = {3.7039, 3.6184, 3.8384};
    heavy_hydrocarbon.eps_k = {150.03, 208.11, 243.87};
    heavy_hydrocarbon.kij = {
        {0.000, 0.012, 0.040},
        {0.012, 0.000, 0.008},
        {0.040, 0.008, 0.000},
    };
    multicomponent_families.push_back(
        {"Heavy HC (3-comp)", heavy_hydrocarbon, 160.0, 520.0});

    for (const auto& family : multicomponent_families) {
        for (int composition_index = 0;
             composition_index < 25;
             ++composition_index) {
            const auto composition = simplex_composition(
                family.parameters.nc, composition_index);
            for (int temperature_index = 0;
                 temperature_index < 150;
                 ++temperature_index) {
                const double temperature = family.temperature_low +
                    (family.temperature_high - family.temperature_low) *
                        temperature_index / 149.0;
                append_state(
                    states,
                    "scaleup_multicomponent",
                    family.name,
                    composition,
                    temperature,
                    1e5,
                    make_mixture_eos(
                        family.parameters, composition, temperature));
            }
        }
    }

    const auto process_parameters = process_18_parameters();
    const std::vector<std::pair<std::string, std::vector<double>>> process_bases = {
        {"Cracked gas (18-comp, light-dominated)",
         {0.15, 0.08, 0.02, 0.005, 0.01, 0.005, 0.003, 0.002,
          0.001, 0.001, 0.001, 0.30, 0.05, 0.02, 0.005, 0.002,
          0.02, 0.01}},
        {"C3 splitter feed (18-comp, C3-focused)",
         {0.005, 0.01, 0.35, 0.005, 0.005, 0.002, 0.001, 0.001,
          0.0, 0.0, 0.0, 0.02, 0.55, 0.01, 0.0, 0.0, 0.005, 0.01}},
        {"Deethanizer overhead (18-comp, C2-rich)",
         {0.20, 0.12, 0.01, 0.0, 0.005, 0.0, 0.0, 0.0, 0.0,
          0.0, 0.0, 0.45, 0.005, 0.0, 0.0, 0.0, 0.05, 0.03}},
    };
    const std::array<std::pair<double, double>, 3> process_temperature_ranges = {
        std::pair{130.0, 420.0},
        std::pair{210.0, 390.0},
        std::pair{130.0, 340.0},
    };
    for (std::size_t family_index = 0;
         family_index < process_bases.size();
         ++family_index) {
        for (int composition_index = 0;
             composition_index < 20;
             ++composition_index) {
            auto composition = process_bases[family_index].second;
            for (std::size_t index = 0; index < composition.size(); ++index) {
                if (composition[index] <= 0.0) {
                    continue;
                }
                composition[index] *= std::exp(
                    0.35 * std::sin(
                        0.73 * (composition_index + 1.0) * (index + 1.0)));
            }
            const double sum = std::accumulate(
                composition.begin(), composition.end(), 0.0);
            for (double& value : composition) {
                value /= sum;
            }
            for (int temperature_index = 0;
                 temperature_index < 120;
                 ++temperature_index) {
                const auto [temperature_low, temperature_high] =
                    process_temperature_ranges[family_index];
                const double temperature = temperature_low +
                    (temperature_high - temperature_low) *
                        temperature_index / 119.0;
                append_state(
                    states,
                    "scaleup_process18",
                    process_bases[family_index].first,
                    composition,
                    temperature,
                    1e5,
                    make_mixture_eos(
                        process_parameters, composition, temperature));
            }
        }
    }

    if (states.size() != 28000) {
        throw std::runtime_error(
            "scale-up isotherm count error: expected 28000, got " +
            std::to_string(states.size()));
    }
    return states;
}

}  // namespace benchmark_states
