#include "direct_root_methods.hpp"

#include <cmath>
#include <cstdlib>
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

stationary_roots::Equation equation(std::vector<double> roots) {
    stationary_roots::Equation result;
    result.pressure = [roots](double x) {
        return polynomial_and_derivative(x, roots).first;
    };
    result.derivative = [roots](double x) {
        return polynomial_and_derivative(x, roots).second;
    };
    result.target_pressure = 0.0;
    result.density_min = 0.5;
    result.density_max = 3.5;
    result.derivative_scale = 10.0;
    return result;
}

void require_roots(
    const direct_roots::MethodResult& result,
    const std::vector<double>& expected) {
    require(result.success, "method must return at least one root");
    require(result.roots.size() == expected.size(), "root count mismatch");
    for (std::size_t index = 0; index < expected.size(); ++index) {
        require(std::abs(result.roots[index].density - expected[index]) < 1e-6,
                "root location mismatch");
        require(std::abs(result.roots[index].residual) < 1e-9,
                "root residual mismatch");
    }
}

}  // namespace

int main() {
    direct_roots::MethodConfig config;
    config.scan_intervals = 8192;
    config.stationary_intervals = 4096;
    config.pressure_tolerance = 1e-10;
    config.density_tolerance = 1e-12;

    const auto cubic = equation({1.0, 2.0, 3.0});
    require_roots(direct_roots::uniform_scan(cubic, config),
                  {1.0, 2.0, 3.0});
    require_roots(direct_roots::stationary_partition(cubic, config),
                  {1.0, 2.0, 3.0});
    require_roots(direct_roots::newton_six_start(cubic, config, false),
                  {1.0, 2.0, 3.0});
    require_roots(direct_roots::newton_six_start(cubic, config, true),
                  {1.0, 2.0, 3.0});

    const auto tangent = equation({1.0, 1.0, 3.0});
    require_roots(direct_roots::stationary_partition(tangent, config),
                  {1.0, 3.0});

    const auto refined = direct_roots::refine_root(cubic, 3.0004, config);
    require(refined.success, "a converged continuation endpoint must refine");
    require(std::abs(refined.root.density - 3.0) < 1e-10,
            "endpoint refinement must recover the target-model root");
    require(std::abs(refined.root.residual) <= config.pressure_tolerance,
            "refined endpoint must satisfy the pressure tolerance");
    require(refined.pressure_evaluations > 0 &&
                refined.derivative_evaluations > 0,
            "endpoint refinement must report its evaluation cost");

    std::cout << "direct root method tests passed\n";
    return 0;
}
