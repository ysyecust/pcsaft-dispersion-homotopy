#include "complete_homotopy_curve.hpp"
#include "benchmark_state_catalog.hpp"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <algorithm>

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        std::exit(1);
    }
}

homotopy::EoSHomotopyInterface linear_curve() {
    homotopy::EoSHomotopyInterface eos;
    eos.T = 300.0;
    eos.rho_min = 0.1;
    eos.rho_max = 3.0;
    eos.rho_ref = 1.0;
    eos.P = [](double rho, double lambda) {
        return rho - 1.0 - lambda;
    };
    eos.dPdrho = [](double, double) { return 1.0; };
    eos.dPdlam = [](double, double) { return -1.0; };
    return eos;
}

homotopy::EoSHomotopyInterface certified_linear_curve() {
    auto eos = linear_curve();
    eos.hard_chain_anchor_is_lowest_root = true;
    eos.dispersion_pressure_strictly_negative = true;
    return eos;
}

double three_crossing_lambda(double rho) {
    return 1.0 + (8.0 / 15.0) *
        (rho - 1.0) * (rho - 2.0) * (rho - 3.0);
}

double three_crossing_derivative(double rho) {
    return (8.0 / 15.0) *
        ((rho - 2.0) * (rho - 3.0) +
         (rho - 1.0) * (rho - 3.0) +
         (rho - 1.0) * (rho - 2.0));
}

homotopy::EoSHomotopyInterface three_crossing_curve() {
    homotopy::EoSHomotopyInterface eos;
    eos.T = 300.0;
    eos.rho_min = 0.1;
    eos.rho_max = 3.8;
    eos.rho_ref = 1.0;
    eos.P = [](double rho, double lambda) {
        return three_crossing_lambda(rho) - lambda;
    };
    eos.dPdrho = [](double rho, double) {
        return three_crossing_derivative(rho);
    };
    eos.dPdlam = [](double, double) { return -1.0; };
    return eos;
}

double tangent_lambda(double rho) {
    return 1.0 + (8.0 / 45.0) *
        (rho - 2.0) * (rho - 2.0) * (rho - 3.0);
}

double tangent_lambda_derivative(double rho) {
    return (8.0 / 45.0) *
        (2.0 * (rho - 2.0) * (rho - 3.0) +
         (rho - 2.0) * (rho - 2.0));
}

homotopy::EoSHomotopyInterface tangent_target_curve() {
    homotopy::EoSHomotopyInterface eos;
    eos.T = 300.0;
    eos.rho_min = 0.1;
    eos.rho_max = 3.8;
    eos.rho_ref = 1.0;
    eos.P = [](double rho, double lambda) {
        return tangent_lambda(rho) - lambda;
    };
    eos.dPdrho = [](double rho, double) {
        return tangent_lambda_derivative(rho);
    };
    eos.dPdlam = [](double, double) { return -1.0; };
    return eos;
}

double narrow_bump_lambda(double rho) {
    const double scaled = (rho - 2.037) / 0.035;
    return -0.08 * (rho - 0.5) + 1.35 * std::exp(-scaled * scaled);
}

double narrow_bump_derivative(double rho) {
    const double scaled = (rho - 2.037) / 0.035;
    return -0.08 - (2.0 * 1.35 / 0.035) *
        scaled * std::exp(-scaled * scaled);
}

homotopy::EoSHomotopyInterface narrow_bump_curve() {
    homotopy::EoSHomotopyInterface eos;
    eos.T = 300.0;
    eos.rho_min = 0.1;
    eos.rho_max = 3.5;
    eos.rho_ref = 1.0;
    eos.P = [](double rho, double lambda) {
        return narrow_bump_lambda(rho) - lambda;
    };
    eos.dPdrho = [](double rho, double) {
        return narrow_bump_derivative(rho);
    };
    eos.dPdlam = [](double, double) { return -1.0; };
    return eos;
}

homotopy::EoSHomotopyInterface high_auxiliary_arch_curve() {
    homotopy::EoSHomotopyInterface eos;
    eos.T = 300.0;
    eos.rho_min = 0.1;
    eos.rho_max = 3.5;
    eos.rho_ref = 1.0;
    eos.P = [](double rho, double lambda) {
        return 1000.0 * (rho - 0.5) * (3.0 - rho) - lambda;
    };
    eos.dPdrho = [](double rho, double) {
        return 1000.0 * (3.5 - 2.0 * rho);
    };
    eos.dPdlam = [](double, double) { return -1.0; };
    return eos;
}

homotopy::CompleteCurveConfig test_config() {
    homotopy::CompleteCurveConfig config;
    config.ds_init = 0.02;
    config.ds_min = 1e-10;
    config.ds_max = 0.08;
    config.residual_tolerance = 1e-11;
    config.event_tolerance = 1e-10;
    config.max_steps_per_direction = 10000;
    config.max_corrector_iterations = 20;
    config.record_path = false;
    return config;
}

void test_bidirectional_trace_reaches_both_density_boundaries() {
    const auto result = homotopy::trace_complete_curve(
        linear_curve(), 0.0, 1.0, test_config());

    require(result.complete,
            "both directional traces must complete on a regular curve");
    require(result.lower_density.reached_density_boundary,
            "negative-density direction must reach the lower boundary");
    require(result.higher_density.reached_density_boundary,
            "positive-density direction must reach the upper boundary");
    require(result.minimum_lambda < -0.85,
            "the auxiliary parameter must be allowed below zero");
    require(result.maximum_lambda > 1.9,
            "the trace must continue after its first lambda=1 crossing");
    require(result.roots.size() == 1,
            "the regular curve has one target-model intersection");
    require(std::abs(result.roots.front().rho - 2.0) < 1e-8,
            "the target-model root must be refined at rho=2");
    require(!result.used_density_scan && !result.used_newton_rescue,
            "the proposed method must not use scanning or Newton rescue");
}

void test_all_three_target_intersections_survive_two_folds() {
    const auto result = homotopy::trace_complete_curve(
        three_crossing_curve(), 0.0, 0.5, test_config());

    require(result.complete,
            "the full cubic graph must be traced to both density boundaries");
    require(result.fold_count >= 2,
            "the traced graph must report both lambda folds");
    require(result.roots.size() == 3,
            "continuation must collect every lambda=1 intersection");
    require(std::abs(result.roots[0].rho - 1.0) < 1e-8,
            "the first target root must equal rho=1");
    require(std::abs(result.roots[1].rho - 2.0) < 1e-8,
            "the second target root must equal rho=2");
    require(std::abs(result.roots[2].rho - 3.0) < 1e-8,
            "the third target root must equal rho=3");
    require(result.roots[0].stable && !result.roots[1].stable &&
                result.roots[2].stable,
            "mechanical stability must follow the target-model derivative");
    for (const auto& root : result.roots) {
        require(std::abs(root.pressure_residual) < 1e-9,
                "each event must be refined on the target model");
    }
}

void test_hard_chain_anchor_requires_no_caller_density_guess() {
    const auto result = homotopy::solve_complete_density_roots(
        linear_curve(), 0.0, test_config());

    require(result.anchor_solved,
            "the public solver must obtain its hard-chain anchor internally");
    require(std::abs(result.anchor_rho - 1.0) < 1e-9,
            "the internally solved hard-chain root must equal rho=1");
    require(result.complete && result.roots.size() == 1,
            "the public solver must return the complete target-root set");
    require(std::abs(result.roots.front().rho - 2.0) < 1e-8,
            "the public solver must refine the target-model root");
}

void test_tangent_target_event_is_not_lost_without_a_sign_change() {
    auto config = test_config();
    config.ds_max = 0.04;
    const auto result = homotopy::solve_complete_density_roots(
        tangent_target_curve(), 0.0, config);

    require(result.complete,
            "the tangent-event curve must be traced to both boundaries");
    if (result.roots.size() != 2) {
        std::cerr << "tangent diagnostic roots=" << result.roots.size();
        for (const auto& root : result.roots) {
            std::cerr << " rho=" << root.rho << " tangent=" << root.tangent;
        }
        std::cerr << '\n';
    }
    require(result.roots.size() == 2,
            "the solver must collect one tangent and one transverse root");
    require(std::abs(result.roots[0].rho - 2.0) < 1e-7,
            "the tangent root must be located at rho=2");
    require(result.roots[0].tangent,
            "the double root must be labelled as a tangent event");
    require(std::abs(result.roots[1].rho - 3.0) < 1e-8,
            "the transverse root must be located at rho=3");
    require(!result.roots[1].tangent,
            "the simple root must remain a transverse event");
}

void test_physical_sign_certificate_terminates_the_root_free_lower_side() {
    const auto result = homotopy::solve_complete_density_roots(
        certified_linear_curve(), 0.0, test_config());

    require(result.complete,
            "a certified root-free lower side must count as complete");
    require(result.lower_density.no_target_root_certified,
            "the lower trace must expose its mathematical termination reason");
    require(!result.lower_density.reached_density_boundary,
            "the certificate must avoid tracing the divergent lower path");
    require(result.lower_density.accepted_steps == 0,
            "the certified lower side must require no continuation steps");
    require(result.higher_density.reached_density_boundary,
            "the root-containing higher side must still reach its boundary");
    require(result.roots.size() == 1 &&
                std::abs(result.roots.front().rho - 2.0) < 1e-8,
            "the certificate must not change the target-root set");
}

void test_narrow_pair_of_target_crossings_is_not_skipped() {
    auto config = test_config();
    config.ds_init = 0.25;
    config.ds_max = 0.35;
    const auto result = homotopy::trace_complete_curve(
        narrow_bump_curve(), 0.0, 0.5, config);
    require(result.complete, "narrow-bump curve must reach both boundaries");
    require(result.roots.size() == 2,
            "one continuation step must not skip a narrow pair of target crossings");
}

void test_large_auxiliary_excursion_does_not_exhaust_arclength_budget() {
    auto config = test_config();
    config.max_steps_per_direction = 2000;
    const auto result = homotopy::trace_complete_curve(
        high_auxiliary_arch_curve(), 0.0, 0.5, config);
    require(result.complete,
            "scaled pseudo-arclength must traverse a large auxiliary excursion");
    require(result.maximum_lambda > 1000.0,
            "regression curve must exercise a genuinely large lambda value");
    require(result.roots.size() == 2,
            "both target crossings around the high auxiliary arch are required");
}

void test_density_direction_does_not_reverse_on_a_sharp_pcsaft_path() {
    const auto states = benchmark_states::build_all_path_states();
    const auto state = std::find_if(
        states.begin(),
        states.end(),
        [](const auto& candidate) {
            return candidate.state_id == "pure_diagnostic_0585";
        });
    require(state != states.end(), "CO2 regression state must exist");

    auto eos = state->eos;
    eos.hard_chain_anchor_is_lowest_root = true;
    eos.dispersion_pressure_strictly_negative = true;
    auto config = test_config();
    config.ds_init = 0.01;
    config.ds_max = 0.05;
    config.residual_tolerance = 1e-9;
    config.event_tolerance = 1e-9;
    config.tangent_angle_target = 0.15;
    config.max_steps_per_direction = 20000;
    const auto result = homotopy::solve_complete_density_roots(
        eos, state->pressure, config);

    if (!result.complete) {
        std::cerr << "CO2 regression diagnostic: higher_failure="
                  << homotopy::to_string(result.higher_density.failure)
                  << ", accepted=" << result.higher_density.accepted_steps
                  << ", rejected=" << result.higher_density.rejected_steps
                  << ", min_lambda=" << result.minimum_lambda
                  << ", max_lambda=" << result.maximum_lambda;
        if (!result.higher_density.path.empty()) {
            const auto& last = result.higher_density.path.back();
            std::cerr << ", last_rbar=" << last.rbar
                      << ", last_lambda=" << last.lambda
                      << ", last_drbar_ds=" << last.drbar_ds
                      << ", last_dlambda_ds=" << last.dlambda_ds
                      << ", last_ds=" << last.ds;
        }
        const double boundary_rho = eos.rho_max *
            (1.0 - config.boundary_relative_margin);
        const double boundary_p0 = eos.P(boundary_rho, 0.0);
        const double boundary_pdisp =
            eos.P(boundary_rho, 1.0) - boundary_p0;
        const double boundary_lambda =
            (state->pressure - boundary_p0) / boundary_pdisp;
        std::cerr << ", boundary_lambda=" << boundary_lambda
                  << ", boundary_Pdisp=" << boundary_pdisp
                  << ", boundary_dPdrho="
                  << eos.dPdrho(boundary_rho, boundary_lambda);
        std::cerr << '\n';
    }
    require(result.complete,
            "the sharp CO2 path must reach the high-density boundary");
    for (std::size_t index = 1;
         index < result.higher_density.path.size();
         ++index) {
        require(result.higher_density.path[index].rbar >=
                    result.higher_density.path[index - 1].rbar,
                "the positive-density trace must never reverse direction");
    }
}

void test_near_fold_pcsaft_root_pair_is_not_skipped() {
    const auto seeds = benchmark_states::build_scaleup_isotherm_seeds();
    const auto seed = std::find_if(
        seeds.begin(),
        seeds.end(),
        [](const auto& candidate) {
            return candidate.state_id == "scaleup_pure_0192";
        });
    require(seed != seeds.end(), "near-fold pure-component seed must exist");
    auto eos = seed->eos;
    eos.hard_chain_anchor_is_lowest_root = true;
    eos.dispersion_pressure_strictly_negative = true;
    auto config = test_config();
    config.residual_tolerance = 1e-9;
    config.event_tolerance = 1e-9;
    config.tangent_lambda_tolerance = 1e-7;
    config.root_merge_relative_tolerance = 1e-7;
    config.ds_init = 0.05;
    config.ds_max = 0.30;
    config.tangent_angle_target = 0.30;
    config.max_steps_per_direction = 20000;
    config.record_path = false;
    const auto result = homotopy::solve_complete_density_roots(
        eos, 3896268.38038543, config);
    require(result.complete, "near-fold PC-SAFT path must complete");
    require(result.roots.size() == 3,
            "near-fold PC-SAFT state must retain all three density roots");
}

void test_pcsaft_event_refinement_does_not_duplicate_a_simple_root() {
    const auto seeds = benchmark_states::build_scaleup_isotherm_seeds();
    const auto seed = std::find_if(
        seeds.begin(),
        seeds.end(),
        [](const auto& candidate) {
            return candidate.state_id == "scaleup_pure_0051";
        });
    require(seed != seeds.end(), "duplicate-root regression seed must exist");
    auto eos = seed->eos;
    eos.hard_chain_anchor_is_lowest_root = true;
    eos.dispersion_pressure_strictly_negative = true;
    auto config = test_config();
    config.residual_tolerance = 1e-9;
    config.event_tolerance = 1e-9;
    config.tangent_lambda_tolerance = 1e-7;
    config.root_merge_relative_tolerance = 1e-7;
    config.ds_init = 0.05;
    config.ds_max = 0.30;
    config.tangent_angle_target = 0.30;
    config.max_steps_per_direction = 20000;
    config.record_path = false;
    const auto result = homotopy::solve_complete_density_roots(
        eos, 1227842.098457237, config);
    require(result.complete, "duplicate-root regression path must complete");
    require(result.roots.size() == 3,
            "event refinement must return each simple PC-SAFT root once");
    for (const auto& root : result.roots) {
        require(std::abs(root.pressure_residual) < 1e-4,
                "each PC-SAFT event root must satisfy the target pressure");
    }
}

void test_pcsaft_event_detection_across_mixture_classes() {
    struct RegressionCase {
        const char* state_id;
        double pressure;
    };
    const std::vector<RegressionCase> cases{
        {"scaleup_binary_1035", 620914.9947268551},
        {"scaleup_multicomponent_10074", 4183898.926858685},
        {"scaleup_process18_20803", 503189.208580068},
    };
    const auto seeds = benchmark_states::build_scaleup_isotherm_seeds();
    for (const auto& regression : cases) {
        const auto seed = std::find_if(
            seeds.begin(),
            seeds.end(),
            [&](const auto& candidate) {
                return candidate.state_id == regression.state_id;
            });
        require(seed != seeds.end(),
                "mixture event-regression seed must exist");
        auto eos = seed->eos;
        eos.hard_chain_anchor_is_lowest_root = true;
        eos.dispersion_pressure_strictly_negative = true;
        auto config = test_config();
        config.residual_tolerance = 1e-9;
        config.event_tolerance = 1e-9;
        config.tangent_lambda_tolerance = 1e-7;
        config.root_merge_relative_tolerance = 1e-7;
    config.ds_init = 0.05;
    config.ds_max = 0.30;
    config.tangent_angle_target = 0.30;
        config.max_steps_per_direction = 20000;
        config.record_path = false;
        const auto result = homotopy::solve_complete_density_roots(
            eos, regression.pressure, config);
        require(result.complete,
                "mixture event-regression path must complete");
        require(result.roots.size() == 3,
                "each mixture regression must return three roots exactly");
    }
}

void test_multifold_pcsaft_curve_retains_four_target_roots() {
    const auto seeds = benchmark_states::build_scaleup_isotherm_seeds();
    const auto seed = std::find_if(
        seeds.begin(),
        seeds.end(),
        [](const auto& candidate) {
            return candidate.state_id == "scaleup_binary_3851";
        });
    require(seed != seeds.end(), "four-root PC-SAFT seed must exist");
    auto eos = seed->eos;
    eos.hard_chain_anchor_is_lowest_root = true;
    eos.dispersion_pressure_strictly_negative = true;
    auto config = test_config();
    config.residual_tolerance = 1e-9;
    config.event_tolerance = 1e-9;
    config.tangent_lambda_tolerance = 1e-7;
    config.root_merge_relative_tolerance = 1e-7;
    config.ds_init = 0.05;
    config.ds_max = 0.30;
    config.tangent_angle_target = 0.30;
    config.max_steps_per_direction = 20000;
    config.record_path = true;
    const auto result = homotopy::solve_complete_density_roots(
        eos, 3737.327060359342, config);
    require(result.complete, "four-root PC-SAFT path must complete");
    require(result.fold_count >= 5,
            "four-root PC-SAFT path must preserve its multifold topology");
    require(result.roots.size() == 4,
            "multifold PC-SAFT curve must return all four target roots");
}

void test_low_pressure_pcsaft_path_survives_large_auxiliary_excursion() {
    const auto seeds = benchmark_states::build_scaleup_isotherm_seeds();
    const auto seed = std::find_if(
        seeds.begin(),
        seeds.end(),
        [](const auto& candidate) {
            return candidate.state_id == "scaleup_binary_4647";
        });
    require(seed != seeds.end(),
            "large-auxiliary PC-SAFT regression seed must exist");
    auto eos = seed->eos;
    eos.hard_chain_anchor_is_lowest_root = true;
    eos.dispersion_pressure_strictly_negative = true;
    auto config = test_config();
    config.residual_tolerance = 1e-9;
    config.event_tolerance = 1e-9;
    config.tangent_lambda_tolerance = 1e-7;
    config.root_merge_relative_tolerance = 1e-7;
    config.ds_init = 0.05;
    config.ds_max = 0.30;
    config.tangent_angle_target = 0.30;
    config.max_steps_per_direction = 20000;
    config.record_path = false;
    const auto result = homotopy::solve_complete_density_roots(
        eos, 18.78303661750248, config);
    require(result.complete,
            "low-pressure PC-SAFT path must complete after large lambda");
    require(result.roots.size() == 3,
            "low-pressure PC-SAFT path must return all three roots");
}

void test_narrow_multicomponent_fold_pair_is_not_skipped() {
    const auto seeds = benchmark_states::build_scaleup_isotherm_seeds();
    const auto seed = std::find_if(
        seeds.begin(),
        seeds.end(),
        [](const auto& candidate) {
            return candidate.state_id == "scaleup_multicomponent_14553";
        });
    require(seed != seeds.end(),
            "narrow multicomponent-fold regression seed must exist");
    auto eos = seed->eos;
    eos.hard_chain_anchor_is_lowest_root = true;
    eos.dispersion_pressure_strictly_negative = true;
    auto config = test_config();
    config.residual_tolerance = 1e-9;
    config.event_tolerance = 1e-9;
    config.tangent_lambda_tolerance = 1e-7;
    config.root_merge_relative_tolerance = 1e-7;
    config.ds_init = 0.05;
    config.ds_max = 0.30;
    config.tangent_angle_target = 0.30;
    config.max_steps_per_direction = 20000;
    config.record_path = false;
    const auto result = homotopy::solve_complete_density_roots(
        eos, 4995405.179621689, config);
    require(result.complete,
            "narrow multicomponent-fold path must complete");
    require(result.roots.size() == 3,
            "narrow multicomponent fold pair must retain all three roots");

    for (const double pressure : {
             4995405.169760037,
             4995406.158833228,
         }) {
        const auto near_stationary =
            homotopy::solve_complete_density_roots(eos, pressure, config);
        require(near_stationary.complete,
                "near-stationary validation path must complete");
        if (near_stationary.roots.size() != 3) {
            std::cerr << "near-stationary diagnostic P=" << pressure
                      << " roots=" << near_stationary.roots.size();
            for (const auto& root : near_stationary.roots) {
                std::cerr << " rho=" << root.rho
                          << " residual=" << root.pressure_residual
                          << " tangent=" << root.tangent;
            }
            std::cerr << '\n';
        }
        require(near_stationary.roots.size() == 3,
                "near-stationary validation pressure must retain three roots");
    }
}

void test_adjacent_event_leaves_do_not_duplicate_pcsaft_roots() {
    const auto seeds = benchmark_states::build_scaleup_isotherm_seeds();
    const auto seed = std::find_if(
        seeds.begin(),
        seeds.end(),
        [](const auto& candidate) {
            return candidate.state_id == "scaleup_multicomponent_13661";
        });
    require(seed != seeds.end(),
            "adjacent-event deduplication regression seed must exist");
    auto eos = seed->eos;
    eos.hard_chain_anchor_is_lowest_root = true;
    eos.dispersion_pressure_strictly_negative = true;
    auto config = test_config();
    config.residual_tolerance = 1e-9;
    config.event_tolerance = 1e-9;
    config.tangent_lambda_tolerance = 1e-7;
    config.root_merge_relative_tolerance = 1e-7;
    config.ds_init = 0.05;
    config.ds_max = 0.30;
    config.tangent_angle_target = 0.30;
    config.max_steps_per_direction = 20000;
    config.record_path = false;
    const auto result = homotopy::solve_complete_density_roots(
        eos, 4944715.294598987, config);
    require(result.complete,
            "adjacent-event deduplication path must complete");
    require(result.roots.size() == 3,
            "adjacent adaptive event leaves must not duplicate a PC-SAFT root");
}

}  // namespace

int main() {
    test_bidirectional_trace_reaches_both_density_boundaries();
    test_all_three_target_intersections_survive_two_folds();
    test_hard_chain_anchor_requires_no_caller_density_guess();
    test_tangent_target_event_is_not_lost_without_a_sign_change();
    test_physical_sign_certificate_terminates_the_root_free_lower_side();
    test_narrow_pair_of_target_crossings_is_not_skipped();
    test_large_auxiliary_excursion_does_not_exhaust_arclength_budget();
    test_density_direction_does_not_reverse_on_a_sharp_pcsaft_path();
    test_near_fold_pcsaft_root_pair_is_not_skipped();
    test_pcsaft_event_refinement_does_not_duplicate_a_simple_root();
    test_pcsaft_event_detection_across_mixture_classes();
    test_multifold_pcsaft_curve_retains_four_target_roots();
    test_low_pressure_pcsaft_path_survives_large_auxiliary_excursion();
    test_narrow_multicomponent_fold_pair_is_not_skipped();
    test_adjacent_event_leaves_do_not_duplicate_pcsaft_roots();
    std::cout << "complete_homotopy_curve tests passed\n";
    return 0;
}
