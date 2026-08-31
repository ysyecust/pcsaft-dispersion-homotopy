// Correctness tests for the reviewer-requested external baselines.
//
// The fixed-point embedding is checked algebraically before any benchmark
// number is trusted: the lambda = 1 endpoint must reproduce the target model
// exactly, the lambda = 0 endpoint must have the anchor as its only root, and
// both partial derivatives must agree with finite differences of the embedded
// pressure.

#include "deflation_root_methods.hpp"
#include "fixed_point_homotopy.hpp"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        std::exit(1);
    }
}

void require_close(
    double actual, double expected, double tolerance,
    const std::string& message) {
    const double scale = std::max({1.0, std::abs(actual), std::abs(expected)});
    if (!(std::abs(actual - expected) <= tolerance * scale)) {
        std::cerr << "FAIL: " << message << " (actual " << actual
                  << ", expected " << expected << ")\n";
        std::exit(1);
    }
}

constexpr double kTargetPressure = 100.0;

// Target model with three density roots at rho = 1, 2, 3 for P = 100,
// and a strictly increasing lambda = 0 reference branch.
double target_pressure_model(double rho) {
    return 1000.0 * (rho - 1.0) * (rho - 2.0) * (rho - 3.0) + kTargetPressure;
}

double target_derivative_model(double rho) {
    return 1000.0 * ((rho - 2.0) * (rho - 3.0) +
                     (rho - 1.0) * (rho - 3.0) +
                     (rho - 1.0) * (rho - 2.0));
}

double reference_branch(double rho) { return 40.0 * rho; }

homotopy::EoSHomotopyInterface cubic_model() {
    homotopy::EoSHomotopyInterface eos;
    eos.T = 300.0;
    eos.rho_min = 0.05;
    eos.rho_max = 4.0;
    eos.rho_ref = 1.0;
    eos.P = [](double rho, double lambda) {
        return reference_branch(rho) +
            lambda * (target_pressure_model(rho) - reference_branch(rho));
    };
    eos.dPdrho = [](double rho, double lambda) {
        return 40.0 + lambda * (target_derivative_model(rho) - 40.0);
    };
    eos.dPdlam = [](double rho, double) {
        return target_pressure_model(rho) - reference_branch(rho);
    };
    return eos;
}

void test_embedding_endpoints() {
    const auto eos = cubic_model();
    const double rbar_anchor = 2.5;
    const auto embedded =
        fixed_point_homotopy::embed(eos, kTargetPressure, rbar_anchor);

    for (double rho = 0.2; rho < 3.9; rho += 0.31) {
        // lambda = 1 must be the untouched target model.
        require_close(
            embedded.P(rho, 1.0), eos.P(rho, 1.0), 1e-12,
            "embedded lambda=1 pressure equals the target model");
        require_close(
            embedded.dPdrho(rho, 1.0), eos.dPdrho(rho, 1.0), 1e-12,
            "embedded lambda=1 derivative equals the target model");
        // lambda = 0 residual must vanish only at the anchor and be linear.
        const double residual_zero = embedded.P(rho, 0.0) - kTargetPressure;
        require(
            (rho < rbar_anchor) == (residual_zero < 0.0) || rho == rbar_anchor,
            "lambda=0 residual changes sign exactly at the anchor");
    }
    require_close(
        embedded.P(rbar_anchor, 0.0), kTargetPressure, 1e-12,
        "anchor solves the lambda=0 problem");
    std::cout << "ok: embedding endpoints\n";
}

void test_embedding_derivatives() {
    const auto eos = cubic_model();
    const auto embedded = fixed_point_homotopy::embed(eos, kTargetPressure, 2.5);
    const double step = 1e-6;
    for (double rho : {0.4, 1.3, 2.2, 3.4}) {
        for (double lambda : {-0.5, 0.0, 0.37, 1.0, 1.8}) {
            const double numerical_rho =
                (embedded.P(rho + step, lambda) -
                 embedded.P(rho - step, lambda)) / (2.0 * step);
            require_close(
                embedded.dPdrho(rho, lambda), numerical_rho, 1e-5,
                "dP/drho matches a central difference");
            const double numerical_lambda =
                (embedded.P(rho, lambda + step) -
                 embedded.P(rho, lambda - step)) / (2.0 * step);
            require_close(
                embedded.dPdlam(rho, lambda), numerical_lambda, 1e-6,
                "dP/dlambda matches a central difference");
        }
    }
    std::cout << "ok: embedding derivatives\n";
}

void test_embedding_is_linear_in_lambda() {
    const auto eos = cubic_model();
    const auto embedded = fixed_point_homotopy::embed(eos, kTargetPressure, 1.7);
    for (double rho : {0.6, 2.4, 3.1}) {
        const double at_zero = embedded.P(rho, 0.0);
        const double at_one = embedded.P(rho, 1.0);
        for (double lambda : {-2.0, 0.25, 0.5, 3.0}) {
            require_close(
                embedded.P(rho, lambda),
                at_zero + lambda * (at_one - at_zero), 1e-12,
                "embedded pressure is affine in lambda");
        }
    }
    std::cout << "ok: embedding affine in lambda\n";
}

// Any root reported by a fixed-point trace must be a genuine root of the
// target model.  Completeness is not asserted here: a single fixed-point
// path is bounded by the poles of its own graph, which is the structural
// property the manuscript reports.
void test_fixed_point_roots_are_genuine() {
    const auto eos = cubic_model();
    using Rule = fixed_point_homotopy::AnchorRule;
    for (Rule rule : {Rule::ideal_gas, Rule::mid_domain, Rule::hard_chain}) {
        const auto result =
            fixed_point_homotopy::solve(eos, kTargetPressure, rule);
        for (const auto& root : result.roots) {
            require_close(
                target_pressure_model(root.rho), kTargetPressure, 1e-6,
                "fixed-point root satisfies the target equation (" +
                    fixed_point_homotopy::anchor_rule_name(rule) + ")");
        }
    }
    const auto multi =
        fixed_point_homotopy::solve_multi_start(eos, kTargetPressure, 6);
    for (const auto& root : multi.roots) {
        require_close(
            target_pressure_model(root.rho), kTargetPressure, 1e-6,
            "multi-start fixed-point root satisfies the target equation");
    }
    require(
        multi.roots.size() >= 1,
        "multi-start fixed-point returns at least one root");
    std::cout << "ok: fixed-point roots are genuine ("
              << multi.roots.size() << " from multi-start)\n";
}

void test_find_and_hide_enumerates_distinct_roots() {
    stationary_roots::Equation equation;
    equation.pressure = [](double rho) { return target_pressure_model(rho); };
    equation.derivative =
        [](double rho) { return target_derivative_model(rho); };
    equation.target_pressure = kTargetPressure;
    equation.density_min = 0.05;
    equation.density_max = 4.0;
    equation.derivative_scale = 1.0;

    direct_roots::MethodConfig config;
    config.pressure_tolerance = 1e-5;
    config.derivative_tolerance = 1e-9;
    config.density_tolerance = 1e-12;
    config.relative_merge_tolerance = 1e-7;

    const auto result = deflation_roots::find_and_hide(equation, config, {});
    require(result.success, "find-and-hide converges on the cubic");
    require(
        result.roots.size() == 3,
        "find-and-hide recovers all three cubic roots, got " +
            std::to_string(result.roots.size()));
    const double expected[3] = {1.0, 2.0, 3.0};
    for (std::size_t index = 0; index < result.roots.size(); ++index) {
        require_close(
            result.roots[index].density, expected[index], 1e-6,
            "find-and-hide root " + std::to_string(index));
    }
    // Deflation must not report the same root twice.
    for (std::size_t i = 0; i + 1 < result.roots.size(); ++i) {
        require(
            result.roots[i + 1].density - result.roots[i].density > 1e-3,
            "find-and-hide roots are distinct");
    }
    std::cout << "ok: find-and-hide enumerates 3 distinct roots\n";
}

void test_find_and_hide_counts_evaluations() {
    stationary_roots::Equation equation;
    equation.pressure = [](double rho) { return target_pressure_model(rho); };
    equation.derivative =
        [](double rho) { return target_derivative_model(rho); };
    equation.target_pressure = kTargetPressure;
    equation.density_min = 0.05;
    equation.density_max = 4.0;
    equation.derivative_scale = 1.0;
    const auto result = deflation_roots::find_and_hide(equation, {}, {});
    require(
        result.pressure_evaluations > 0 && result.derivative_evaluations > 0,
        "find-and-hide reports its model evaluations");
    std::cout << "ok: find-and-hide evaluation accounting\n";
}

}  // namespace

int main() {
    test_embedding_endpoints();
    test_embedding_derivatives();
    test_embedding_is_linear_in_lambda();
    test_fixed_point_roots_are_genuine();
    test_find_and_hide_enumerates_distinct_roots();
    test_find_and_hide_counts_evaluations();
    std::cout << "all external-baseline tests passed\n";
    return 0;
}
