#include "reference_root_isolator.hpp"

#include <algorithm>
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

stationary_roots::Equation make_polynomial(
    std::vector<double> roots,
    double lower,
    double upper) {
    stationary_roots::Equation equation;
    equation.pressure = [roots](double x) {
        return polynomial_and_derivative(x, roots).first;
    };
    equation.derivative = [roots](double x) {
        return polynomial_and_derivative(x, roots).second;
    };
    equation.target_pressure = 0.0;
    equation.density_min = lower;
    equation.density_max = upper;
    equation.derivative_scale = 100.0;
    return equation;
}

void require_resolved_roots(const std::vector<double>& roots) {
    auto expected = roots;
    expected.erase(std::unique(expected.begin(), expected.end()), expected.end());
    reference_roots::ReferenceConfig config;
    config.level_intervals = {256, 1024, 4096};
    config.independent_scan_intervals = 16384;
    config.pressure_tolerance = 1e-10;
    config.density_agreement_tolerance = 1e-7;
    const auto result = reference_roots::isolate(
        make_polynomial(roots, 0.5, roots.back() + 0.5), config);
    if (result.status != reference_roots::Status::resolved) {
        std::cerr << "diagnostic root counts:";
        for (int count : result.level_root_counts) {
            std::cerr << ' ' << count;
        }
        std::cerr << "; stationary counts:";
        for (int count : result.level_stationary_counts) {
            std::cerr << ' ' << count;
        }
        std::cerr << '\n';
    }
    require(result.status == reference_roots::Status::resolved,
            "reference hierarchy must resolve the " +
                std::to_string(roots.size()) + "-factor polynomial: " +
                result.reason);
    require(result.roots.size() == expected.size(),
            "resolved polynomial root count mismatch");
    require(!result.independent_roots.empty(),
            "independent scan must retain its diagnostic roots");
    for (std::size_t index = 0; index < expected.size(); ++index) {
        require(std::abs(result.roots[index].density - expected[index]) < 1e-6,
                "resolved polynomial root location mismatch");
    }
}

}  // namespace

int main() {
    require_resolved_roots({1.0});
    require_resolved_roots({1.0, 2.0, 3.0});
    require_resolved_roots({1.0, 2.0, 3.0, 4.0, 5.0});
    require_resolved_roots({1.0, 1.0, 3.0});
    require_resolved_roots({0.999, 1.001, 2.0});

    reference_roots::ReferenceConfig coarse;
    coarse.level_intervals = {4, 8, 16};
    coarse.independent_scan_intervals = 20000;
    coarse.pressure_tolerance = 1e-10;
    coarse.density_agreement_tolerance = 1e-7;
    const auto unresolved = reference_roots::isolate(
        make_polynomial({1.0, 1.01, 1.02}, 0.5, 2.0), coarse);
    require(unresolved.status == reference_roots::Status::unresolved,
            "under-resolved hierarchy must be labeled unresolved");

    std::cout << "reference root isolator tests passed\n";
    return 0;
}
