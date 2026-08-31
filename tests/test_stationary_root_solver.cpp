#include "stationary_root_solver.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        std::exit(1);
    }
}

std::pair<double, double> polynomial_and_derivative(
    double x,
    const std::vector<double>& roots) {
    double value = 1.0;
    for (double root : roots) {
        value *= x - root;
    }

    double derivative = 0.0;
    for (std::size_t omitted = 0; omitted < roots.size(); ++omitted) {
        double term = 1.0;
        for (std::size_t index = 0; index < roots.size(); ++index) {
            if (index != omitted) {
                term *= x - roots[index];
            }
        }
        derivative += term;
    }
    return {value, derivative};
}

stationary_roots::SolverConfig strict_config() {
    stationary_roots::SolverConfig config;
    config.linear_intervals = 4096;
    config.log_intervals = 1024;
    config.pressure_tolerance = 1e-10;
    config.derivative_tolerance = 1e-10;
    config.density_tolerance = 1e-12;
    config.relative_merge_tolerance = 1e-9;
    return config;
}

void require_roots(
    const stationary_roots::SolveResult& result,
    const std::vector<double>& expected,
    double tolerance = 1e-7) {
    require(result.success, "solver must report success");
    require(result.roots.size() == expected.size(),
            "root count differs from the expected count");
    for (std::size_t index = 0; index < expected.size(); ++index) {
        require(std::abs(result.roots[index].density - expected[index]) <
                    tolerance,
                "root density differs from the expected value");
        require(std::abs(result.roots[index].residual) < 1e-9,
                "root pressure residual exceeds tolerance");
    }
}

void test_monotone_equation_has_one_root_and_no_stationary_points() {
    stationary_roots::Equation equation;
    equation.pressure = [](double density) { return density * density; };
    equation.derivative = [](double density) { return 2.0 * density; };
    equation.target_pressure = 4.0;
    equation.density_min = 0.1;
    equation.density_max = 3.0;
    equation.derivative_scale = 6.0;

    const auto result = stationary_roots::solve(equation, strict_config());

    require_roots(result, {2.0});
    require(result.stationary_points.empty(),
            "monotone equation must not contain stationary points");
    require(result.roots.front().mechanically_stable,
            "positive derivative root must be mechanically stable");
}

void test_cubic_equation_has_two_stationary_points_and_three_roots() {
    const std::vector<double> expected = {1.0, 2.0, 3.0};
    stationary_roots::Equation equation;
    equation.pressure = [expected](double density) {
        return polynomial_and_derivative(density, expected).first;
    };
    equation.derivative = [expected](double density) {
        return polynomial_and_derivative(density, expected).second;
    };
    equation.target_pressure = 0.0;
    equation.density_min = 0.5;
    equation.density_max = 3.5;
    equation.derivative_scale = 10.0;

    const auto result = stationary_roots::solve(equation, strict_config());

    require_roots(result, expected);
    require(result.stationary_points.size() == 2,
            "cubic equation must contain two stationary points");
    require(result.roots[0].mechanically_stable &&
                !result.roots[1].mechanically_stable &&
                result.roots[2].mechanically_stable,
            "cubic stability labels must alternate positive/negative/positive");
}

void test_quintic_equation_returns_all_five_roots() {
    const std::vector<double> expected = {1.0, 2.0, 3.0, 4.0, 5.0};
    stationary_roots::Equation equation;
    equation.pressure = [expected](double density) {
        return polynomial_and_derivative(density, expected).first;
    };
    equation.derivative = [expected](double density) {
        return polynomial_and_derivative(density, expected).second;
    };
    equation.target_pressure = 0.0;
    equation.density_min = 0.5;
    equation.density_max = 5.5;
    equation.derivative_scale = 100.0;

    const auto result = stationary_roots::solve(equation, strict_config());

    require_roots(result, expected);
    require(result.stationary_points.size() == 4,
            "quintic equation must contain four stationary points");
}

void test_even_multiplicity_root_is_detected_at_a_stationary_point() {
    stationary_roots::Equation equation;
    equation.pressure = [](double density) {
        return (density - 1.0) * (density - 1.0) * (density - 3.0);
    };
    equation.derivative = [](double density) {
        return (density - 1.0) * (3.0 * density - 7.0);
    };
    equation.target_pressure = 0.0;
    equation.density_min = 0.5;
    equation.density_max = 3.5;
    equation.derivative_scale = 20.0;

    const auto result = stationary_roots::solve(equation, strict_config());

    require_roots(result, {1.0, 3.0});
    require(result.roots.front().at_stationary_point,
            "double root must be identified as a stationary-point root");
    require(!result.roots.front().mechanically_stable,
            "zero-derivative tangent root must not be labeled mechanically stable");
}

void test_nearby_roots_are_not_merged_by_a_percent_scale_tolerance() {
    constexpr double epsilon = 1e-6;
    const double separation = std::sqrt(epsilon);
    stationary_roots::Equation equation;
    equation.pressure = [](double density) {
        const double shifted = density - 1.0;
        return (density - 2.0) * (shifted * shifted - epsilon);
    };
    equation.derivative = [](double density) {
        const double shifted = density - 1.0;
        return shifted * shifted - epsilon +
               2.0 * (density - 2.0) * shifted;
    };
    equation.target_pressure = 0.0;
    equation.density_min = 0.5;
    equation.density_max = 2.5;
    equation.derivative_scale = 10.0;

    const auto result = stationary_roots::solve(equation, strict_config());

    require_roots(result, {1.0 - separation, 1.0 + separation, 2.0}, 2e-7);
}

void test_steep_high_density_root_is_refined_to_pressure_tolerance() {
    const std::vector<double> expected = {
        3.3690551183492,
        1411.7852820382,
        33172.262500357,
    };
    stationary_roots::Equation equation;
    equation.pressure = [expected](double density) {
        return 1e-3 * polynomial_and_derivative(density, expected).first;
    };
    equation.derivative = [expected](double density) {
        return 1e-3 * polynomial_and_derivative(density, expected).second;
    };
    equation.target_pressure = 0.0;
    equation.density_min = 1e-3;
    equation.density_max = 35000.0;
    equation.derivative_scale = 2e6;

    auto config = strict_config();
    config.pressure_tolerance = 1e-5;
    config.derivative_tolerance = 1e-8;
    const auto result = stationary_roots::solve(equation, config);

    require(result.roots.size() == 3,
            "steep high-density root must not be rejected after bracketing");
    for (std::size_t index = 0; index < expected.size(); ++index) {
        require(std::abs(result.roots[index].density - expected[index]) < 1e-6,
                "steep-root density mismatch");
        require(std::abs(result.roots[index].residual) < 1e-4,
                "steep-root residual must be refined after bisection");
    }
}

}  // namespace

int main() {
    test_monotone_equation_has_one_root_and_no_stationary_points();
    test_cubic_equation_has_two_stationary_points_and_three_roots();
    test_quintic_equation_returns_all_five_roots();
    test_even_multiplicity_root_is_detected_at_a_stationary_point();
    test_nearby_roots_are_not_merged_by_a_percent_scale_tolerance();
    test_steep_high_density_root_is_refined_to_pressure_tolerance();
    std::cout << "stationary root solver tests passed\n";
    return 0;
}
