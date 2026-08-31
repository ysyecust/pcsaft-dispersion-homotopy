// ============================================================
// embedding_pole_audit.cpp
//
// Structural comparison of the two homotopy embeddings on real catalogue
// isotherms.  Both embeddings are affine in lambda, so each has an explicit
// solution graph
//
//   lambda(rho) = [P_target - P(rho, 0)] / [P(rho, 1) - P(rho, 0)],
//
// whose poles are the zeros of the denominator.  A pole disconnects the graph
// and therefore bounds the set of target roots any single trace can reach.
//
//   dispersion-strength embedding : denominator = P_disp(rho)
//   fixed-point embedding         : denominator = F(rho) - S (rbar - rbar_0)
//                                   with F = P(rho,1) - P_target
//
// The audit counts sign changes of each denominator on a dense mesh, and
// tests the prediction that the fixed-point denominator must change sign
// whenever target roots lie on both sides of the anchor.
// ============================================================

#include "all_root_state_catalog.hpp"
#include "complete_homotopy_curve.hpp"
#include "fixed_point_homotopy.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace {

constexpr int kMeshIntervals = 20000;

int count_sign_changes(const std::vector<double>& values) {
    int changes = 0;
    for (std::size_t index = 0; index + 1 < values.size(); ++index) {
        const double left = values[index];
        const double right = values[index + 1];
        if (!std::isfinite(left) || !std::isfinite(right)) {
            continue;
        }
        if ((left < 0.0 && right > 0.0) || (left > 0.0 && right < 0.0)) {
            ++changes;
        }
    }
    return changes;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        std::size_t limit_per_group = 40;
        std::filesystem::path output_directory = "pole_audit";
        for (int index = 1; index + 1 < argc; index += 2) {
            const std::string option = argv[index];
            if (option == "--output") {
                output_directory = argv[index + 1];
            } else if (option == "--limit-per-group") {
                limit_per_group =
                    static_cast<std::size_t>(std::stoull(argv[index + 1]));
            }
        }
        std::filesystem::create_directories(output_directory);
        std::ofstream rows(output_directory / "cr_pole_audit.csv");
        rows << std::setprecision(16);
        rows << "state_id,group,target_class,reference_root_count,"
                "dispersion_denominator_sign_changes,"
                "anchor_rule,anchor_rho,roots_below_anchor,roots_above_anchor,"
                "fixed_point_denominator_sign_changes,"
                "prediction_holds,criterion_crossings,criterion_minimum,"
                "criterion_scan_evaluations\n";

        const auto formal = all_root_states::validation_config();
        long long states_total = 0;
        long long dispersion_poles_total = 0;
        long long fixed_point_with_pole = 0;
        long long fixed_point_cases = 0;
        long long prediction_cases = 0;
        long long prediction_failures = 0;

        for (int group = 0; group < 4; ++group) {
            auto chunk = formal;
            chunk.pure_states = group == 0 ? formal.pure_states : 0;
            chunk.binary_states = group == 1 ? formal.binary_states : 0;
            chunk.multicomponent_states =
                group == 2 ? formal.multicomponent_states : 0;
            chunk.process18_states = group == 3 ? formal.process18_states : 0;
            const auto catalog = all_root_states::build(chunk);
            const std::size_t available = catalog.states.size();
            const std::size_t take = std::min(limit_per_group, available);
            for (std::size_t selected = 0; selected < take; ++selected) {
                const std::size_t position = take == 1
                    ? available / 2
                    : static_cast<std::size_t>(std::llround(
                          static_cast<double>(selected) /
                          static_cast<double>(take - 1) *
                          static_cast<double>(available - 1)));
                const auto& generated = catalog.states[position];
                const auto& state = generated.state;
                const auto& eos = state.eos;
                const double target = state.pressure;

                const double lower = eos.rho_min * (1.0 + 1e-10);
                const double upper = eos.rho_max * (1.0 - 1e-10);
                std::vector<double> mesh(kMeshIntervals + 1);
                std::vector<double> dispersion(kMeshIntervals + 1);
                std::vector<double> residual(kMeshIntervals + 1);
                for (int index = 0; index <= kMeshIntervals; ++index) {
                    const double fraction = static_cast<double>(index) /
                        static_cast<double>(kMeshIntervals);
                    const double rho = lower + fraction * (upper - lower);
                    mesh[index] = rho;
                    const double hard_chain = eos.P(rho, 0.0);
                    const double full = eos.P(rho, 1.0);
                    dispersion[index] = full - hard_chain;
                    residual[index] = full - target;
                }
                const int dispersion_changes = count_sign_changes(dispersion);
                dispersion_poles_total += dispersion_changes;

                // Approximate target roots from residual sign changes.
                std::vector<double> roots;
                for (int index = 0; index < kMeshIntervals; ++index) {
                    if ((residual[index] < 0.0 && residual[index + 1] > 0.0) ||
                        (residual[index] > 0.0 && residual[index + 1] < 0.0)) {
                        roots.push_back(
                            0.5 * (mesh[index] + mesh[index + 1]));
                    }
                }

                const double pressure_scale =
                    homotopy::complete_curve_detail::pressure_reference(
                        eos, target);
                // Aslam and Sunol starting-point criterion, applied over the
                // admissible interval.
                homotopy::CompleteCurveConfig criterion_config;
                const auto criterion_scan =
                    fixed_point_homotopy::select_anchor_by_criterion(
                        eos, target, criterion_config);

                using Rule = fixed_point_homotopy::AnchorRule;
                for (Rule rule :
                     {Rule::ideal_gas, Rule::mid_domain, Rule::hard_chain}) {
                    double rbar_anchor = 0.0;
                    int anchor_evaluations = 0;
                    homotopy::CompleteCurveConfig config;
                    if (!fixed_point_homotopy::select_anchor(
                            eos, target, rule, config,
                            rbar_anchor, anchor_evaluations)) {
                        continue;
                    }
                    const double anchor_rho = rbar_anchor * eos.rho_ref;
                    std::vector<double> denominator(kMeshIntervals + 1);
                    for (int index = 0; index <= kMeshIntervals; ++index) {
                        denominator[index] = residual[index] -
                            pressure_scale *
                                (mesh[index] / eos.rho_ref - rbar_anchor);
                    }
                    const int changes = count_sign_changes(denominator);
                    int below = 0;
                    int above = 0;
                    for (double root : roots) {
                        if (root < anchor_rho) {
                            ++below;
                        } else {
                            ++above;
                        }
                    }
                    const bool straddles = below > 0 && above > 0;
                    const bool prediction_holds = !straddles || changes >= 1;
                    if (straddles) {
                        ++prediction_cases;
                        if (!prediction_holds) {
                            ++prediction_failures;
                        }
                    }
                    ++fixed_point_cases;
                    if (changes >= 1) {
                        ++fixed_point_with_pole;
                    }
                    rows << state.state_id << ','
                         << all_root_states::group_name(state) << ','
                         << difficult_states::target_class_name(
                                generated.target_class)
                         << ',' << roots.size() << ',' << dispersion_changes
                         << ',' << fixed_point_homotopy::anchor_rule_name(rule)
                         << ',' << anchor_rho << ',' << below << ',' << above
                         << ',' << changes << ',' << (prediction_holds ? 1 : 0)
                         << ',' << criterion_scan.crossing_count
                         << ',' << criterion_scan.minimum_achievable
                         << ',' << criterion_scan.state_evaluations << '\n';
                }
                ++states_total;
            }
        }

        std::ofstream report(output_directory / "cr_pole_audit_report.txt");
        report << "Embedding pole audit\n"
               << "states audited: " << states_total << '\n'
               << "dispersion-graph denominator sign changes (total): "
               << dispersion_poles_total << '\n'
               << "fixed-point anchor cases: " << fixed_point_cases << '\n'
               << "fixed-point cases with at least one pole: "
               << fixed_point_with_pole << '\n'
               << "cases where roots straddle the anchor: "
               << prediction_cases << '\n'
               << "prediction failures (straddle without a pole): "
               << prediction_failures << '\n';
        report.close();
        std::cout << "states audited: " << states_total
                  << ", dispersion poles " << dispersion_poles_total
                  << ", fixed-point with pole " << fixed_point_with_pole
                  << "/" << fixed_point_cases
                  << ", straddle cases " << prediction_cases
                  << ", prediction failures " << prediction_failures << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "embedding_pole_audit: " << error.what() << '\n';
        return 1;
    }
}
