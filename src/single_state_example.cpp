// ============================================================
// single_state_example.cpp
//
// Smallest useful entry point into the solver: pick one state from the
// catalogue, solve it, and print the classified root set next to the
// independently computed reference roots.  Runs in a few seconds.
//
//   ./single_state_example                 # a default three-root state
//   ./single_state_example --index 1234    # any catalogue state
// ============================================================

#include "all_root_state_catalog.hpp"
#include "complete_homotopy_curve.hpp"
#include "reference_root_isolator.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>

int main(int argc, char** argv) {
    std::size_t index = 7;
    for (int argument = 1; argument + 1 < argc; argument += 2) {
        if (std::string(argv[argument]) == "--index") {
            index = static_cast<std::size_t>(std::stoull(argv[argument + 1]));
        }
    }

    const auto catalog = all_root_states::build(all_root_states::quick_config());
    if (catalog.states.empty()) {
        std::fprintf(stderr, "empty catalogue\n");
        return 1;
    }
    const auto& generated = catalog.states[index % catalog.states.size()];
    const auto& state = generated.state;

    std::printf("state      : %s\n", state.state_id.c_str());
    std::printf("T          : %.4f K\n", state.temperature);
    std::printf("P_target   : %.6e Pa\n", state.pressure);
    std::printf("density domain : (%.6e, %.6e) mol/m^3\n\n",
                state.eos.rho_min, state.eos.rho_max);

    auto eos = state.eos;
    eos.hard_chain_anchor_is_lowest_root = true;
    eos.dispersion_pressure_strictly_negative = true;

    homotopy::CompleteCurveConfig config;
    config.ds_init = 0.05;
    config.ds_min = 1e-10;
    config.ds_max = 0.30;
    config.residual_tolerance = 1e-9;
    config.tangent_angle_target = 0.30;
    config.max_steps_per_direction = 20000;
    config.max_evaluations = 100000;

    const auto solved =
        homotopy::solve_complete_density_roots(eos, state.pressure, config);

    std::printf("anchor rho : %.10e mol/m^3\n", solved.anchor_rho);
    std::printf("path complete (both traces reached the domain boundary): %s\n",
                solved.complete ? "yes" : "no");
    if (!solved.complete) {
        std::printf("failure class: %s\n",
                    homotopy::to_string(solved.higher_density.failure));
    }
    std::printf("regular folds traversed: %d\n\n", solved.fold_count);

    std::printf("returned roots\n");
    std::printf("  %-22s %-16s %-10s %s\n",
                "density [mol/m^3]", "residual [Pa]", "dP/drho", "class");
    for (const auto& root : solved.roots) {
        std::printf("  %-22.12e %-16.3e %-10.3e %s\n",
                    root.rho, root.pressure_residual, root.dPdrho,
                    root.tangent ? "marginal"
                                 : (root.stable ? "stable" : "unstable"));
    }

    // Independent check: stationary-point hierarchy on the same isotherm.
    stationary_roots::Equation equation;
    equation.pressure = [&](double rho) { return state.eos.P(rho, 1.0); };
    equation.derivative = [&](double rho) { return state.eos.dPdrho(rho, 1.0); };
    equation.target_pressure = state.pressure;
    equation.density_min = state.eos.rho_min;
    equation.density_max = state.eos.rho_max * (1.0 - 1e-10);
    // The reference hierarchy normalizes its derivative tolerance by this
    // scale; leaving it at one makes the stationary-point search miss turns.
    {
        const double middle =
            0.5 * (equation.density_min + equation.density_max);
        equation.derivative_scale = std::max({
            1.0,
            std::abs(equation.derivative(equation.density_min)),
            std::abs(equation.derivative(middle)),
            std::abs(equation.derivative(equation.density_max)),
        });
    }

    reference_roots::ReferenceConfig reference_config;
    reference_config.level_intervals = {128, 512, 2048};
    reference_config.independent_scan_intervals = 4096;
    reference_config.pressure_tolerance = 1e-5;
    reference_config.derivative_tolerance = 1e-9;
    reference_config.density_tolerance = 1e-12;
    reference_config.density_agreement_tolerance = 1e-5;
    reference_config.root_merge_tolerance = 1e-9;
    const auto reference = reference_roots::isolate(equation, reference_config);
    if (reference.status != reference_roots::Status::resolved) {
        std::printf("\nreference hierarchy did not resolve this state\n");
        return 1;
    }

    std::printf("\nindependent reference roots (stationary-point hierarchy)\n");
    for (const auto& root : reference.roots) {
        std::printf("  %-22.12e %-16.3e %s\n", root.density, root.residual,
                    root.mechanically_stable ? "stable" : "unstable");
    }
    std::printf("\nreturned %zu roots, reference %zu roots -> %s\n",
                solved.roots.size(), reference.roots.size(),
                solved.roots.size() == reference.roots.size()
                    ? "counts agree" : "COUNTS DISAGREE");
    return 0;
}
