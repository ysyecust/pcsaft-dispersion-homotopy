// ============================================================
// external_baseline_benchmark.cpp
//
// Reviewer-requested comparison against the two published methods that
// target the same problem:
//
//   [25] Aslam and Sunol (2006), global fixed-point homotopy
//        -> fixed_point_homotopy_{ideal_gas,hard_chain,mid_domain}
//   [26] Monroy-Loperena (2026), find-and-hide
//        -> find_and_hide_deflation           (deflation reading)
//        -> stationary_partition_{512,8192}   (partition reading)
//
// The run also reports, for every method, completeness measured on the
// mechanically stable roots alone, decomposed by target class, so that a
// missing stable root is separated from a missing intermediate root.
//
// Every method is evaluated on the same frozen catalogue, against the
// same independently verified reference roots, with the same evaluation
// counter and the same root-matching tolerance.
// ============================================================

#include "all_root_state_catalog.hpp"
#include "complete_homotopy_curve.hpp"
#include "deflation_root_methods.hpp"
#include "direct_root_methods.hpp"
#include "fixed_point_homotopy.hpp"
#include "matched_completeness_benchmark.hpp"
#include "reference_root_isolator.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;
using ReferenceMap = std::unordered_map<
    std::string, std::vector<solver_benchmark::Root>>;

enum class MethodKind {
    newton_single,
    uniform_scan,
    newton_six,
    newton_six_safeguarded,
    stationary_partition,
    proposed,
    fixed_point,
    find_and_hide,
};

struct MethodSpec {
    MethodKind kind;
    std::string name;
    int budget = 0;
    fixed_point_homotopy::AnchorRule anchor =
        fixed_point_homotopy::AnchorRule::ideal_gas;
};

struct MethodOutput {
    bool success = false;
    std::string failure;
    std::vector<solver_benchmark::Root> roots;
    matched_completeness::EvaluationCounter counter;
    long long wall_time_ns = 0;
};

struct Aggregate {
    long long states = 0;
    long long complete_all = 0;
    long long complete_stable = 0;
    long long complete_stable_tau[3] = {0, 0, 0};
    long long matched = 0;
    long long missed = 0;
    long long extra = 0;
    long long missed_stable = 0;
    long long extra_stable = 0;
    long long membership_mismatch_states = 0;
    long long total_calls = 0;
    long long total_wall_time_ns = 0;
    std::vector<double> wall_times_ns;
    std::vector<double> scalar_calls;
};

struct ClassAggregate {
    long long states = 0;
    long long complete_all = 0;
    long long complete_stable = 0;
    long long missed_stable = 0;
    long long extra_stable = 0;
};

std::string csv(const std::string& value) {
    if (value.find_first_of(",\"\r\n") == std::string::npos) {
        return value;
    }
    std::string escaped = "\"";
    for (char character : value) {
        escaped += character == '"' ? "\"\"" : std::string(1, character);
    }
    return escaped + '"';
}

stationary_roots::Equation equation_for(
    const benchmark_states::StateDefinition& state,
    const homotopy::EoSHomotopyInterface& eos,
    bool evaluate_derivative_scale = false) {
    stationary_roots::Equation equation;
    equation.pressure = [&](double density) { return eos.P(density, 1.0); };
    equation.derivative =
        [&](double density) { return eos.dPdrho(density, 1.0); };
    equation.target_pressure = state.pressure;
    equation.density_min = eos.rho_min;
    equation.density_max = eos.rho_max * (1.0 - 1e-10);
    equation.derivative_scale = 1.0;
    if (evaluate_derivative_scale) {
        const double middle =
            0.5 * (equation.density_min + equation.density_max);
        equation.derivative_scale = std::max({
            1.0,
            std::abs(equation.derivative(equation.density_min)),
            std::abs(equation.derivative(middle)),
            std::abs(equation.derivative(equation.density_max)),
        });
    }
    return equation;
}

std::vector<solver_benchmark::Root> convert(
    const std::vector<direct_roots::RootRecord>& roots) {
    std::vector<solver_benchmark::Root> converted;
    converted.reserve(roots.size());
    for (const auto& root : roots) {
        converted.push_back({
            root.density, root.residual, root.derivative,
            root.mechanically_stable, false,
        });
    }
    return converted;
}

std::vector<solver_benchmark::Root> convert(
    const std::vector<homotopy::RootEvent>& roots) {
    std::vector<solver_benchmark::Root> converted;
    converted.reserve(roots.size());
    for (const auto& root : roots) {
        converted.push_back({
            root.rho, root.pressure_residual, root.dPdrho,
            root.stable, root.tangent,
        });
    }
    return converted;
}

std::vector<solver_benchmark::Root> convert(
    const reference_roots::Result& result) {
    std::vector<solver_benchmark::Root> converted;
    converted.reserve(result.roots.size());
    for (const auto& root : result.roots) {
        converted.push_back({
            root.density, root.residual, root.derivative,
            root.mechanically_stable, root.at_stationary_point,
        });
    }
    return converted;
}

// tau values for the mechanical-stability sensitivity study
constexpr double kChiTolerances[3] = {1e-10, 1e-9, 1e-8};

// Optional diagnostics: restrict the run to the tangent target classes and
// dump every reference and returned root with its chi indicator, so that the
// stability-label agreement can be evaluated offline at any threshold.
bool g_only_tangent = false;
std::ofstream g_root_dump;

inline void fill_chi(
    std::vector<solver_benchmark::Root>& roots, double pressure_scale) {
    for (auto& root : roots) {
        root.chi = std::isfinite(root.derivative)
            ? root.density * root.derivative / pressure_scale
            : 0.0;
    }
}

ReferenceMap load_references(const std::filesystem::path& path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error(
            "cannot open reference roots: " + path.string());
    }
    ReferenceMap references;
    std::string line;
    std::getline(input, line);
    while (std::getline(input, line)) {
        const auto parsed =
            matched_completeness::parse_reference_root_row(line);
        if (parsed.valid) {
            references[parsed.state_id].push_back(parsed.root);
        }
    }
    for (auto& [state_id, roots] : references) {
        std::sort(
            roots.begin(), roots.end(),
            [](const auto& lhs, const auto& rhs) {
                return lhs.density < rhs.density;
            });
    }
    return references;
}

reference_roots::ReferenceConfig quick_reference_config() {
    reference_roots::ReferenceConfig config;
    config.level_intervals = {128, 512, 2048};
    config.independent_scan_intervals = 4096;
    config.pressure_tolerance = 1e-5;
    config.derivative_tolerance = 1e-9;
    config.density_tolerance = 1e-12;
    config.density_agreement_tolerance = 1e-5;
    config.root_merge_tolerance = 1e-9;
    return config;
}

homotopy::CompleteCurveConfig proposed_config() {
    homotopy::CompleteCurveConfig config;
    config.ds_init = 0.05;
    config.ds_min = 1e-10;
    config.ds_max = 0.30;
    config.residual_tolerance = 1e-9;
    config.event_tolerance = 1e-9;
    config.tangent_lambda_tolerance = 1e-7;
    config.root_merge_relative_tolerance = 1e-7;
    config.tangent_angle_target = 0.30;
    config.max_steps_per_direction = 20000;
    config.max_evaluations = 100000;
    config.record_path = false;
    return config;
}

direct_roots::MethodConfig direct_config(int budget) {
    direct_roots::MethodConfig config;
    config.scan_intervals = budget;
    config.stationary_intervals = budget;
    config.pressure_tolerance = 1e-5;
    config.derivative_tolerance = 1e-9;
    config.density_tolerance = 1e-12;
    config.relative_merge_tolerance = 1e-7;
    return config;
}

std::vector<MethodSpec> methods() {
    using Rule = fixed_point_homotopy::AnchorRule;
    return {
        {MethodKind::newton_single, "newton_single_ideal_gas", 1},
        {MethodKind::uniform_scan, "uniform_scan_512", 512},
        {MethodKind::uniform_scan, "uniform_scan_8192", 8192},
        {MethodKind::newton_six, "newton_six", 6},
        {MethodKind::newton_six_safeguarded, "newton_six_safeguarded", 6},
        {MethodKind::stationary_partition, "stationary_partition_512", 512},
        {MethodKind::stationary_partition, "stationary_partition_8192", 8192},
        {MethodKind::find_and_hide, "find_and_hide_deflation", 6},
        {MethodKind::fixed_point,
         "fixed_point_homotopy_criterion", 0, Rule::criterion},
        {MethodKind::fixed_point,
         "fixed_point_homotopy_hard_chain", 0, Rule::hard_chain},
        {MethodKind::fixed_point,
         "fixed_point_homotopy_multi_start_log12", -12, Rule::multi_start},
        {MethodKind::proposed, "dispersion_homotopy_pseudo_arclength", 0},
    };
}

MethodOutput run_method(
    const benchmark_states::StateDefinition& state,
    const MethodSpec& method) {
    MethodOutput output;
    auto eos = matched_completeness::counted_eos(state.eos, output.counter);
    const auto started = Clock::now();

    if (method.kind == MethodKind::proposed) {
        eos.hard_chain_anchor_is_lowest_root = true;
        eos.dispersion_pressure_strictly_negative = true;
        const auto solved = homotopy::solve_complete_density_roots(
            eos, state.pressure, proposed_config());
        output.success = solved.complete;
        output.failure = solved.complete
            ? "none"
            : homotopy::to_string(
                  solved.anchor_solved ? solved.higher_density.failure
                                       : solved.anchor_failure);
        output.roots = convert(solved.roots);
    } else if (method.kind == MethodKind::newton_single) {
        const auto equation = equation_for(state, eos);
        constexpr double gas_constant = 8.31446261815324;
        const double initial_density = state.pressure /
            std::max(1e-30, gas_constant * state.temperature);
        const auto solved = direct_roots::refine_root(
            equation, initial_density, direct_config(1));
        output.success = solved.success;
        output.failure = solved.success ? "none" : "no_converged_root";
        if (solved.success) {
            output.roots = convert(
                std::vector<direct_roots::RootRecord>{solved.root});
        }
    } else if (method.kind == MethodKind::fixed_point) {
        const auto solved =
            method.anchor == fixed_point_homotopy::AnchorRule::multi_start
                ? fixed_point_homotopy::solve_multi_start(
                      eos, state.pressure, std::abs(method.budget),
                      proposed_config(), method.budget < 0)
                : fixed_point_homotopy::solve(
                      eos, state.pressure, method.anchor, proposed_config());
        output.success = solved.complete;
        output.failure = solved.complete
            ? "none" : homotopy::to_string(solved.failure);
        output.roots = convert(solved.roots);
    } else if (method.kind == MethodKind::find_and_hide) {
        const auto equation = equation_for(state, eos);
        const auto solved = deflation_roots::find_and_hide(
            equation, direct_config(method.budget), {});
        output.success = solved.success;
        output.failure =
            solved.failure_reason.empty() ? "none" : solved.failure_reason;
        output.roots = convert(solved.roots);
    } else {
        const bool needs_derivative_scale =
            method.kind == MethodKind::stationary_partition;
        const auto equation =
            equation_for(state, eos, needs_derivative_scale);
        const auto config = direct_config(method.budget);
        direct_roots::MethodResult solved;
        if (method.kind == MethodKind::newton_six) {
            solved = direct_roots::newton_six_start(equation, config, false);
        } else if (method.kind == MethodKind::newton_six_safeguarded) {
            solved = direct_roots::newton_six_start(equation, config, true);
        } else if (method.kind == MethodKind::uniform_scan) {
            solved = direct_roots::uniform_scan(equation, config);
        } else {
            solved = direct_roots::stationary_partition(equation, config);
        }
        output.success = solved.success;
        output.failure =
            solved.failure_reason.empty() ? "none" : solved.failure_reason;
        output.roots = convert(solved.roots);
    }

    output.wall_time_ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            Clock::now() - started).count();
    return output;
}

std::vector<std::size_t> selected_indices(
    std::size_t size, std::size_t limit) {
    if (limit == 0 || limit >= size) {
        std::vector<std::size_t> indices(size);
        for (std::size_t index = 0; index < size; ++index) {
            indices[index] = index;
        }
        return indices;
    }
    std::vector<std::size_t> indices;
    indices.reserve(limit);
    for (std::size_t selected = 0; selected < limit; ++selected) {
        const double fraction = limit == 1
            ? 0.5
            : static_cast<double>(selected) /
                  static_cast<double>(limit - 1);
        indices.push_back(static_cast<std::size_t>(
            std::llround(fraction * static_cast<double>(size - 1))));
    }
    return indices;
}

void write_summary(
    const std::filesystem::path& output_directory,
    const std::map<std::string, Aggregate>& aggregates,
    const std::map<std::string, ClassAggregate>& class_aggregates) {
    std::ofstream summary(output_directory / "cr_method_summary.csv");
    std::ofstream classes(output_directory / "cr_class_summary.csv");
    std::ofstream report(output_directory / "cr_report.txt");
    summary << std::setprecision(16);
    classes << std::setprecision(16);
    summary << "method,states,complete_all,complete_all_fraction,"
               "complete_stable,complete_stable_fraction,matched_roots,"
               "missed_roots,extra_roots,missed_stable_roots,"
               "extra_stable_roots,complete_stable_tau1e-10,"
               "complete_stable_tau1e-9,complete_stable_tau1e-8,"
               "total_scalar_calls,mean_scalar_calls,"
               "median_scalar_calls,median_wall_time_ns,p99_wall_time_ns,"
               "total_wall_time_s,correct_results_per_s,"
               "stable_membership_mismatch_states\n";
    for (const auto& [method, aggregate] : aggregates) {
        const double states = static_cast<double>(std::max(
            aggregate.states, 1LL));
        const double seconds = aggregate.total_wall_time_ns * 1e-9;
        summary << method << ',' << aggregate.states << ','
                << aggregate.complete_all << ','
                << aggregate.complete_all / states << ','
                << aggregate.complete_stable << ','
                << aggregate.complete_stable / states << ','
                << aggregate.matched << ',' << aggregate.missed << ','
                << aggregate.extra << ',' << aggregate.missed_stable << ','
                << aggregate.extra_stable << ','
                << aggregate.complete_stable_tau[0] << ','
                << aggregate.complete_stable_tau[1] << ','
                << aggregate.complete_stable_tau[2] << ','
                << aggregate.total_calls << ','
                << aggregate.total_calls / states << ','
                << matched_completeness::quantile(aggregate.scalar_calls, 0.5)
                << ','
                << matched_completeness::quantile(aggregate.wall_times_ns, 0.5)
                << ','
                << matched_completeness::quantile(aggregate.wall_times_ns, 0.99)
                << ',' << seconds << ','
                << (seconds > 0.0 ? aggregate.complete_all / seconds : 0.0)
                << ',' << aggregate.membership_mismatch_states << '\n';
        report << method << ": complete_all " << aggregate.complete_all
               << '/' << aggregate.states << ", complete_stable "
               << aggregate.complete_stable << ", missed_stable "
               << aggregate.missed_stable << ", extra_stable "
               << aggregate.extra_stable << ", mean scalar calls "
               << aggregate.total_calls / states << '\n';
    }
    classes << "method,target_class,states,complete_all,complete_stable,"
               "complete_stable_fraction,missed_stable_roots,"
               "extra_stable_roots\n";
    for (const auto& [key, aggregate] : class_aggregates) {
        const auto separator = key.find('|');
        classes << key.substr(0, separator) << ','
                << key.substr(separator + 1) << ',' << aggregate.states << ','
                << aggregate.complete_all << ',' << aggregate.complete_stable
                << ','
                << static_cast<double>(aggregate.complete_stable) /
                       static_cast<double>(std::max(aggregate.states, 1LL))
                << ',' << aggregate.missed_stable << ','
                << aggregate.extra_stable << '\n';
    }
}

void run(
    bool quick,
    bool validation_catalog,
    const std::filesystem::path& reference_path,
    const std::filesystem::path& output_directory,
    std::size_t limit_per_group) {
    std::filesystem::create_directories(output_directory);
    std::ofstream rows(output_directory / "cr_state_method_results.csv");
    if (!rows) {
        throw std::runtime_error("cannot create result CSV");
    }
    rows << std::setprecision(16);
    rows << "state_id,group,target_class,T_K,P_Pa,reference_roots,method,"
            "returned_roots,matched_roots,missed_roots,extra_roots,"
            "missed_stable_roots,extra_stable_roots,complete_all_roots,"
            "complete_stable_roots,total_scalar_calls,wall_time_ns,"
            "failure_reason\n";

    ReferenceMap references;
    if (!quick) {
        references = load_references(reference_path);
    }
    std::vector<difficult_states::CatalogConfig> configs;
    if (quick) {
        configs.push_back(all_root_states::quick_config());
    } else {
        const auto formal = validation_catalog
            ? all_root_states::validation_config()
            : all_root_states::formal_config();
        for (int group = 0; group < 4; ++group) {
            auto chunk = formal;
            chunk.pure_states = group == 0 ? formal.pure_states : 0;
            chunk.binary_states = group == 1 ? formal.binary_states : 0;
            chunk.multicomponent_states =
                group == 2 ? formal.multicomponent_states : 0;
            chunk.process18_states = group == 3 ? formal.process18_states : 0;
            configs.push_back(std::move(chunk));
        }
    }

    std::map<std::string, Aggregate> aggregates;
    std::map<std::string, ClassAggregate> class_aggregates;
    std::size_t processed = 0;
    const auto method_list = methods();
    for (const auto& config : configs) {
        const auto catalog = all_root_states::build(config);
        const auto validation = all_root_states::validate(catalog, config);
        if (!validation.valid) {
            throw std::runtime_error(
                "catalogue validation failed: " + validation.reason);
        }
        const auto indices = selected_indices(
            catalog.states.size(), quick ? 0 : limit_per_group);
        for (std::size_t local = 0; local < indices.size(); ++local) {
            const auto& generated = catalog.states[indices[local]];
            const auto& state = generated.state;
            std::vector<solver_benchmark::Root> reference;
            if (quick) {
                const auto reference_result = reference_roots::isolate(
                    equation_for(state, state.eos, true),
                    quick_reference_config());
                if (reference_result.status !=
                    reference_roots::Status::resolved) {
                    throw std::runtime_error(
                        "quick reference unresolved: " + state.state_id);
                }
                reference = convert(reference_result);
            } else {
                const auto found = references.find(state.state_id);
                if (found == references.end()) {
                    throw std::runtime_error(
                        "reference missing: " + state.state_id);
                }
                reference = found->second;
            }

            auto ordered_methods = method_list;
            std::rotate(
                ordered_methods.begin(),
                ordered_methods.begin() +
                    static_cast<std::ptrdiff_t>(
                        processed % ordered_methods.size()),
                ordered_methods.end());
            const std::string class_name =
                difficult_states::target_class_name(generated.target_class);
            const double pressure_scale =
                homotopy::complete_curve_detail::pressure_reference(
                    state.eos, state.pressure);
            if (g_only_tangent && class_name == "interior_three_root") {
                ++processed;
                continue;
            }
            fill_chi(reference, pressure_scale);
            if (g_root_dump.is_open()) {
                for (std::size_t i = 0; i < reference.size(); ++i) {
                    const auto& root = reference[i];
                    g_root_dump << csv(state.state_id) << ',' << class_name
                                << ",reference," << i << ',' << root.density
                                << ',' << root.chi << ',' << root.residual
                                << ',' << root.derivative << '\n';
                }
            }
            for (const auto& method : ordered_methods) {
                auto solved = run_method(state, method);
                fill_chi(solved.roots, pressure_scale);
                if (g_root_dump.is_open()) {
                    for (std::size_t i = 0; i < solved.roots.size(); ++i) {
                        const auto& root = solved.roots[i];
                        g_root_dump << csv(state.state_id) << ',' << class_name
                                    << ',' << method.name << ',' << i << ','
                                    << root.density << ',' << root.chi << ','
                                    << root.residual << ',' << root.derivative
                                    << '\n';
                    }
                }
                auto comparison = solver_benchmark::compare_roots(
                    reference, solved.roots, 2e-6);
                // Primary stable-set metrics: stable subsets at the
                // reference threshold tau_chi = 1e-9, compared by density.
                const auto primary = solver_benchmark::compare_stable_subsets(
                    reference, solved.roots, 2e-6, kChiTolerances[1]);
                comparison.missed_stable = primary.missed_stable;
                comparison.extra_stable = primary.extra_stable;
                comparison.complete_stable = primary.complete;
                bool exact_stable_tau[3];
                for (int t = 0; t < 3; ++t) {
                    const auto at_tau = solver_benchmark::compare_stable_subsets(
                        reference, solved.roots, 2e-6, kChiTolerances[t]);
                    exact_stable_tau[t] = at_tau.complete;
                }
                rows << csv(state.state_id) << ','
                     << all_root_states::group_name(state) << ','
                     << class_name << ',' << state.temperature << ','
                     << state.pressure << ',' << reference.size() << ','
                     << method.name << ',' << solved.roots.size() << ','
                     << comparison.matched << ',' << comparison.missed << ','
                     << comparison.extra << ',' << comparison.missed_stable
                     << ',' << comparison.extra_stable << ','
                     << comparison.complete_all << ','
                     << comparison.complete_stable << ','
                     << solved.counter.total() << ',' << solved.wall_time_ns
                     << ',' << csv(solved.failure) << '\n';

                auto& aggregate = aggregates[method.name];
                ++aggregate.states;
                aggregate.complete_all += comparison.complete_all ? 1 : 0;
                aggregate.complete_stable +=
                    comparison.complete_stable ? 1 : 0;
                for (int t = 0; t < 3; ++t) {
                    aggregate.complete_stable_tau[t] +=
                        exact_stable_tau[t] ? 1 : 0;
                }
                aggregate.matched += comparison.matched;
                aggregate.missed += comparison.missed;
                aggregate.extra += comparison.extra;
                aggregate.missed_stable += comparison.missed_stable;
                aggregate.extra_stable += comparison.extra_stable;
                aggregate.membership_mismatch_states +=
                    primary.membership_mismatches > 0 ? 1 : 0;
                aggregate.total_calls += solved.counter.total();
                aggregate.total_wall_time_ns += solved.wall_time_ns;
                aggregate.wall_times_ns.push_back(
                    static_cast<double>(solved.wall_time_ns));
                aggregate.scalar_calls.push_back(
                    static_cast<double>(solved.counter.total()));

                auto& per_class =
                    class_aggregates[method.name + "|" + class_name];
                ++per_class.states;
                per_class.complete_all += comparison.complete_all ? 1 : 0;
                per_class.complete_stable +=
                    comparison.complete_stable ? 1 : 0;
                per_class.missed_stable += comparison.missed_stable;
                per_class.extra_stable += comparison.extra_stable;
            }
            ++processed;
            if (!quick && processed % 500 == 0) {
                std::cout << "processed states: " << processed << '\n'
                          << std::flush;
            }
        }
    }
    write_summary(output_directory, aggregates, class_aggregates);
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc == 3 && std::string(argv[1]) == "--quick-test") {
            run(true, false, {}, argv[2], 0);
            return 0;
        }
        if (argc >= 5 && std::string(argv[1]) == "--reference" &&
            std::string(argv[3]) == "--output") {
            bool validation_catalog = false;
            std::size_t limit = 0;
            for (int index = 5; index < argc; index += 2) {
                if (std::string(argv[index]) != "--only-tangent" && index + 1 >= argc) {
                    throw std::runtime_error("option needs a value");
                }
                const std::string option = argv[index];
                if (option == "--catalog") {
                    const std::string value = argv[index + 1];
                    if (value != "development" && value != "validation") {
                        throw std::runtime_error(
                            "catalog must be development or validation");
                    }
                    validation_catalog = value == "validation";
                } else if (option == "--limit-per-group") {
                    limit = static_cast<std::size_t>(
                        std::stoull(argv[index + 1]));
                } else if (option == "--only-tangent") {
                    g_only_tangent = true;
                    --index;  // flag without a value
                } else if (option == "--dump-roots") {
                    g_root_dump.open(argv[index + 1]);
                    if (!g_root_dump) {
                        throw std::runtime_error("cannot open root dump");
                    }
                    g_root_dump << std::setprecision(17)
                                << "state_id,target_class,method,root_index,"
                                   "density,chi,residual,derivative\n";
                } else {
                    throw std::runtime_error("unknown option: " + option);
                }
            }
            run(false, validation_catalog, argv[2], argv[4], limit);
            return 0;
        }
        std::cerr << "usage: external_baseline_benchmark --quick-test OUT\n"
                     "   or: external_baseline_benchmark --reference CSV "
                     "--output DIR [--catalog development|validation] "
                     "[--limit-per-group N] [--only-tangent] [--dump-roots FILE]\n";
        return 2;
    } catch (const std::exception& error) {
        std::cerr << "external_baseline_benchmark: " << error.what() << '\n';
        return 1;
    }
}
