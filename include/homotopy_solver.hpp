#pragma once
// ============================================================
// homotopy_solver.hpp — Physical Parameter Homotopy Continuation
//
// Pseudo arc-length continuation in non-dimensionalized (rbar, lam)
// coordinates for robust EoS density solving.
//
// Supports:
//   - Dispersion-strength homotopy (PC-SAFT: lam scales Z_disp)
//   - Association-strength homotopy (CPA: lam scales eps_assoc)
//   - Any EoS via FunctionInterface
//   - Homotopy-assisted stable-root collection at lam=1
// ============================================================

#include <cmath>
#include <vector>
#include <functional>
#include <algorithm>
#include <cstdio>
#include <limits>

namespace homotopy {

enum class HomotopyFailure {
    none,
    invalid_anchor,
    singular_tangent,
    corrector_failure,
    step_below_minimum,
    maximum_steps,
    evaluation_budget,
    domain_exit,
};

inline const char* to_string(HomotopyFailure failure) {
    switch (failure) {
        case HomotopyFailure::none: return "none";
        case HomotopyFailure::invalid_anchor: return "invalid_anchor";
        case HomotopyFailure::singular_tangent: return "singular_tangent";
        case HomotopyFailure::corrector_failure: return "corrector_failure";
        case HomotopyFailure::step_below_minimum: return "step_below_minimum";
        case HomotopyFailure::maximum_steps: return "maximum_steps";
        case HomotopyFailure::evaluation_budget: return "evaluation_budget";
        case HomotopyFailure::domain_exit: return "domain_exit";
    }
    return "unknown";
}

// ---- EoS function interface ----
struct EoSEvaluation {
    double P;
    double dPdrho;
    double dPdlam;
};

struct EoSHomotopyInterface {
    std::function<double(double rho, double lambda)> P;
    std::function<double(double rho, double lambda)> dPdrho;
    std::function<double(double rho, double lambda)> dPdlam;
    // Optional fused callback.  PC-SAFT implementations can reuse model
    // intermediates and avoid repeating the same state calculation.
    std::function<EoSEvaluation(double rho, double lambda)> evaluate_all;

    double rho_min;
    double rho_max;
    double rho_ref;
    double T;

    // Optional model-level certificates.  They are false for a generic EoS
    // and may only be enabled when established for the stated model domain.
    bool hard_chain_anchor_is_lowest_root = false;
    bool dispersion_pressure_strictly_negative = false;
};

// ---- Solver configuration ----
struct HomotopyConfig {
    double ds_init   = 0.05;
    double ds_min    = 1e-12;
    double ds_max    = 2.0;
    double tol       = 1e-8;
    int    max_steps = 5000;   // increased from 2000
    int    max_evals = std::numeric_limits<int>::max();
    int    max_corr  = 30;
    double angle_target = 0.4;
    bool   verbose   = false;
    bool   record_path = false;
    bool   no_rescue = false;  // if true, skip Newton/lambda-stepping rescue
};

// ---- Result ----
struct PathPoint { double rbar; double lam; double P_residual; double ds; };

struct FoldState {
    double rbar;
    double lam;
    double drbar_ds;
    double dlam_ds;
    double ds;
};

struct HomotopyResult {
    std::vector<double> roots;
    std::vector<bool>   stable;
    int    total_steps    = 0;
    int    total_evals    = 0;
    int    n_folds        = 0;
    bool   success        = false;
    HomotopyFailure failure = HomotopyFailure::none;
    std::vector<PathPoint> path;
    std::vector<FoldState> fold_states;
};

// ---- Non-dimensionalized residual ----
struct NDResidual {
    double H;
    double H_rbar;
    double H_lam;
};

inline NDResidual eval_nd(const EoSHomotopyInterface& eos,
                           double rbar, double lam,
                           double P_target, double P_ref) {
    double rho = rbar * eos.rho_ref;
    double P;
    double dPr;
    double dPl;
    if (eos.evaluate_all) {
        const auto value = eos.evaluate_all(rho, lam);
        P = value.P;
        dPr = value.dPdrho;
        dPl = value.dPdlam;
    } else {
        P = eos.P(rho, lam);
        dPr = eos.dPdrho(rho, lam);
        dPl = eos.dPdlam(rho, lam);
    }

    return {
        (P - P_target) / P_ref,
        dPr * eos.rho_ref / P_ref,
        dPl / P_ref
    };
}

// ============================================================
// Core: Pseudo Arc-Length Continuation (improved v2)
//
// Key improvements:
//   - Higher max_steps (5000) for deep VLE loops at low T
//   - Smarter step recovery after corrector failures
//   - Stall detection with ds boost
//   - Relaxed convergence for near-fold conditions
//   - Saves fold states for spinodal diagnostics
// ============================================================

inline HomotopyResult continuation(const EoSHomotopyInterface& eos,
                                    double P_target,
                                    double rbar_anchor,
                                    const HomotopyConfig& cfg = {}) {
    HomotopyResult result;

    constexpr double R_GAS = 8.314462;
    double P_scale_eos = eos.rho_ref * R_GAS * eos.T;
    double P_ref = std::max(std::abs(P_target), P_scale_eos * 0.01);
    if (P_ref < 1.0) P_ref = 1.0;

    double rbar = rbar_anchor;
    double lam  = 0.0;

    bool evaluation_budget_exhausted = false;
    auto evaluate = [&](double rbar_value, double lambda_value) {
        if (result.total_evals >= cfg.max_evals) {
            evaluation_budget_exhausted = true;
            const double nan = std::numeric_limits<double>::quiet_NaN();
            return NDResidual{nan, nan, nan};
        }
        ++result.total_evals;
        return eval_nd(
            eos, rbar_value, lambda_value, P_target, P_ref);
    };

    auto nd = evaluate(rbar, lam);
    if (evaluation_budget_exhausted) {
        result.failure = HomotopyFailure::evaluation_budget;
        return result;
    }

    if (std::abs(nd.H) > 0.01) {
        for (int i = 0; i < 50; ++i) {
            if (std::abs(nd.H) < cfg.tol) break;
            if (std::abs(nd.H_rbar) < 1e-30) break;
            double delta = nd.H / nd.H_rbar;
            if (std::abs(delta) > 0.3 * rbar) delta = 0.3 * rbar * (delta > 0 ? 1 : -1);
            rbar -= delta;
            rbar = std::max(1e-8, std::min(rbar, eos.rho_max / eos.rho_ref * 0.98));
            nd = evaluate(rbar, lam);
            if (evaluation_budget_exhausted) {
                result.failure = HomotopyFailure::evaluation_budget;
                return result;
            }
        }
    }

    if (std::abs(nd.H) > 1e-4) {
        if (cfg.verbose) std::printf("  [HOMOTOPY] Anchor solve failed: |H| = %.2e\n", std::abs(nd.H));
        result.failure = HomotopyFailure::invalid_anchor;
        return result;
    }

    double norm = std::sqrt(nd.H_lam * nd.H_lam + nd.H_rbar * nd.H_rbar);
    if (norm < 1e-30) {
        result.failure = HomotopyFailure::singular_tangent;
        return result;
    }

    double drbar_ds = -nd.H_lam / norm;
    double dlam_ds  =  nd.H_rbar / norm;

    if (dlam_ds < 0) { drbar_ds = -drbar_ds; dlam_ds = -dlam_ds; }

    double ds = cfg.ds_init;

    if (cfg.record_path) {
        double rho = rbar * eos.rho_ref;
        double P_res = eos.P(rho, lam) - P_target;
        result.path.push_back({rbar, lam, P_res, ds});
        result.total_evals++;
    }

    double prev_dlam_ds = dlam_ds;
    int consecutive_failures = 0;
    double lam_at_last_progress = 0.0;
    int steps_since_lam_progress = 0;

    // Track max lambda reached for Newton rescue
    double lam_max_reached = 0.0;
    double rbar_at_lam_max = rbar;

    for (int step = 0; step < cfg.max_steps; ++step) {
        result.total_steps++;

        // Stall detection
        if (std::abs(lam - lam_at_last_progress) > 0.02) {
            lam_at_last_progress = lam;
            steps_since_lam_progress = 0;
        } else {
            steps_since_lam_progress++;
        }

        if (steps_since_lam_progress > 80 && ds < cfg.ds_init * 2.0) {
            ds = std::min(cfg.ds_max, ds * 4.0);
            steps_since_lam_progress = 0;
        }

        // Predictor
        double rbar_pred = rbar + ds * drbar_ds;
        double lam_pred  = lam  + ds * dlam_ds;

        if (lam_pred > 1.0) {
            if (std::abs(ds * dlam_ds) > 1e-30) {
                double scale = (1.0 - lam) / (ds * dlam_ds);
                rbar_pred = rbar + ds * scale * drbar_ds;
            }
            lam_pred = 1.0;
        }
        if (lam_pred < 0.0) lam_pred = 0.0;

        double rbar_max = eos.rho_max / eos.rho_ref * 0.98;
        rbar_pred = std::max(1e-8, std::min(rbar_pred, rbar_max));

        // Corrector
        double rbar_c = rbar_pred;
        double lam_c  = lam_pred;
        bool converged = false;

        for (int corr = 0; corr < cfg.max_corr; ++corr) {
            auto ndc = evaluate(rbar_c, lam_c);
            if (evaluation_budget_exhausted) {
                result.failure = HomotopyFailure::evaluation_budget;
                return result;
            }

            double rhs1 = -ndc.H;
            double rhs2 = -((rbar_c - rbar_pred) * drbar_ds +
                            (lam_c - lam_pred) * dlam_ds);

            double det = ndc.H_rbar * dlam_ds - ndc.H_lam * drbar_ds;
            double det_scale = std::max({std::abs(ndc.H_rbar), std::abs(ndc.H_lam), 1e-20});
            if (std::abs(det) < 1e-15 * det_scale) break;

            double d_rbar = (rhs1 * dlam_ds  - ndc.H_lam * rhs2) / det;
            double d_lam  = (ndc.H_rbar * rhs2 - rhs1 * drbar_ds) / det;

            double max_d = 0.5;
            if (std::abs(d_rbar) > max_d) d_rbar = max_d * (d_rbar > 0 ? 1 : -1);
            if (std::abs(d_lam) > max_d)  d_lam  = max_d * (d_lam > 0 ? 1 : -1);

            rbar_c += d_rbar;
            lam_c  += d_lam;

            rbar_c = std::max(1e-8, std::min(rbar_c, rbar_max));
            lam_c  = std::max(0.0, std::min(lam_c, 1.0));

            if (std::abs(ndc.H) < cfg.tol &&
                std::abs(d_rbar) < cfg.tol * 100 &&
                std::abs(d_lam) < cfg.tol * 100) {
                converged = true;
                break;
            }
        }

        if (!converged) {
            ds *= 0.25;
            consecutive_failures++;
            if (ds < cfg.ds_min || consecutive_failures > 10) {  // matches paper Algorithm 1 cap
                if (cfg.verbose)
                    std::printf("  [HOMOTOPY] Step %d: ds below minimum at lam=%.4f\n",
                                step, lam);
                result.failure = ds < cfg.ds_min
                    ? HomotopyFailure::step_below_minimum
                    : HomotopyFailure::corrector_failure;
                break;
            }
            continue;
        }
        consecutive_failures = 0;

        // Update tangent
        auto ndc = evaluate(rbar_c, lam_c);
        if (evaluation_budget_exhausted) {
            result.failure = HomotopyFailure::evaluation_budget;
            return result;
        }

        norm = std::sqrt(ndc.H_lam * ndc.H_lam + ndc.H_rbar * ndc.H_rbar);
        if (norm < 1e-30) {
            result.failure = HomotopyFailure::singular_tangent;
            break;
        }

        double new_drbar_ds = -ndc.H_lam / norm;
        double new_dlam_ds  =  ndc.H_rbar / norm;

        if (new_drbar_ds * drbar_ds + new_dlam_ds * dlam_ds < 0) {
            new_drbar_ds = -new_drbar_ds;
            new_dlam_ds  = -new_dlam_ds;
        }

        // Fold detection
        if (prev_dlam_ds * new_dlam_ds < 0) {
            result.n_folds++;
            result.fold_states.push_back({rbar_c, lam_c, new_drbar_ds, new_dlam_ds, ds});
            if (cfg.verbose)
                std::printf("  [HOMOTOPY] Fold point at lam=%.4f, rbar=%.4f (step %d)\n",
                            lam_c, rbar_c, step);
        }

        // Step size adaptation
        double cos_angle = drbar_ds * new_drbar_ds + dlam_ds * new_dlam_ds;
        cos_angle = std::max(-1.0, std::min(1.0, cos_angle));
        double angle = std::acos(cos_angle);

        double ratio = cfg.angle_target / std::max(angle, 1e-10);
        ratio = std::max(0.1, std::min(5.0, ratio));
        ds = std::max(cfg.ds_min, std::min(cfg.ds_max, ds * ratio));

        rbar = rbar_c;
        lam  = lam_c;
        drbar_ds = new_drbar_ds;
        dlam_ds  = new_dlam_ds;
        prev_dlam_ds = new_dlam_ds;

        // Track maximum lambda reached
        if (lam > lam_max_reached) {
            lam_max_reached = lam;
            rbar_at_lam_max = rbar;
        }

        if (cfg.record_path) {
            double rho_tmp = rbar * eos.rho_ref;
            double P_res = eos.P(rho_tmp, lam) - P_target;
            result.path.push_back({rbar, lam, P_res, ds});
            result.total_evals++;
        }

        if (lam >= 1.0 - cfg.tol) {
            double rho_root = rbar * eos.rho_ref;
            result.roots.push_back(rho_root);

            double dPdr = eos.dPdrho(rho_root, 1.0);
            result.total_evals++;
            result.stable.push_back(dPdr > 0);

            if (cfg.verbose)
                std::printf("  [HOMOTOPY] Root found: rho = %.4f mol/m3 (%s)\n",
                            rho_root, dPdr > 0 ? "stable" : "unstable");

            result.success = true;
            result.failure = HomotopyFailure::none;
            break;
        }
    }

    if (!result.success && result.failure == HomotopyFailure::none) {
        result.failure = HomotopyFailure::maximum_steps;
    }

    // ---- Newton rescue: if continuation failed, try Newton at lam=1 ----
    // The continuation may have failed because:
    // 1. The gas root disappears (fold) before lam=1 -> need liquid root
    // 2. The path is too winding -> try direct Newton from multiple starts
    if (!result.success && !cfg.no_rescue) {
        constexpr double R_GAS_RESCUE = 8.314462;
        double rho_ideal = P_target / (R_GAS_RESCUE * eos.T);

        // Try multiple starting points at lam=1
        double starts[] = {
            rbar_at_lam_max * eos.rho_ref,  // best point from continuation
            rho_ideal,                       // ideal gas
            eos.rho_max * 0.3,
            eos.rho_max * 0.5,
            eos.rho_max * 0.7,
            eos.rho_max * 0.85,
        };

        for (auto rho0 : starts) {
            if (rho0 <= 0) continue;
            double rho_try = rho0;
            for (int i = 0; i < 50; ++i) {
                double P_val = eos.P(rho_try, 1.0);
                double dPdr = eos.dPdrho(rho_try, 1.0);
                result.total_evals += 2;
                double f = P_val - P_target;
                if (std::abs(f) < 1.0) {
                    result.roots.push_back(rho_try);
                    result.stable.push_back(dPdr > 0);
                    result.success = true;
                    if (cfg.verbose)
                        std::printf("  [HOMOTOPY] Newton rescue: rho=%.4f (from rho0=%.4f)\n",
                                    rho_try, rho0);
                    break;
                }
                if (std::abs(dPdr) < 1e-10) break;
                double rho_new = rho_try - f / dPdr;
                if (rho_new <= 0 || rho_new > eos.rho_max * 1.1) break;
                rho_try = rho_new;
            }
            if (result.success) break;
        }

        // Lambda-stepping rescue from highest lambda reached
        if (!result.success && lam_max_reached > 0.3) {
            double rho_step = rbar_at_lam_max * eos.rho_ref;
            double lam_step = lam_max_reached;
            double dlam_target = 0.05;

            while (lam_step < 1.0 - 1e-8) {
                lam_step = std::min(lam_step + dlam_target, 1.0);

                bool step_ok = false;
                for (int i = 0; i < 30; ++i) {
                    double P_val = eos.P(rho_step, lam_step);
                    double dPdr = eos.dPdrho(rho_step, lam_step);
                    result.total_evals += 2;
                    double f = P_val - P_target;
                    if (std::abs(f) < 1.0) { step_ok = true; break; }
                    if (std::abs(dPdr) < 1e-10) break;
                    double rho_new = rho_step - f / dPdr;
                    if (rho_new <= 0 || rho_new > eos.rho_max * 1.1) break;
                    rho_step = rho_new;
                }

                if (!step_ok) {
                    dlam_target *= 0.5;
                    lam_step -= dlam_target * 2;
                    if (dlam_target < 1e-6) break;
                    continue;
                }

                if (lam_step >= 1.0 - 1e-8) {
                    result.roots.push_back(rho_step);
                    double dPdr = eos.dPdrho(rho_step, 1.0);
                    result.total_evals++;
                    result.stable.push_back(dPdr > 0);
                    result.success = true;
                    if (cfg.verbose)
                        std::printf("  [HOMOTOPY] Lambda-stepping rescue: rho=%.4f\n", rho_step);
                    break;
                }
            }
        }
    }

    return result;
}

// ============================================================
// Homotopy-assisted stable-root collection
//
// Strategy: trace the hard-chain anchor branch to lam=1, then use the
// lambda-invariant PC-SAFT feasibility domain to bracket remaining
// target-EoS roots. This avoids simulator-supplied density intervals
// while keeping the implementation honest about its internal
// feasibility-domain scan.
// ============================================================

inline HomotopyResult continuation_multi_branch(const EoSHomotopyInterface& eos,
                                                  double P_target,
                                                  double rbar_anchor,
                                                  const HomotopyConfig& cfg = {}) {
    // Step 1: Run continuation from gas-like anchor
    auto result = continuation(eos, P_target, rbar_anchor, cfg);

    // This interface returns mechanically stable roots only.  The lower-level
    // continuation routine retains the endpoint stability tag because it is
    // also used to diagnose complete homotopy paths.
    {
        std::vector<double> stable_roots;
        for (size_t i = 0; i < result.roots.size(); ++i) {
            if (i < result.stable.size() && result.stable[i]) {
                stable_roots.push_back(result.roots[i]);
            }
        }
        result.roots = std::move(stable_roots);
        result.stable.assign(result.roots.size(), true);
    }

    // Step 2: Find the remaining stable roots at lam=1 via density scanning
    // Use adaptive resolution: coarser in gas region, finer near packing
    // The liquid root is in a very steep region (dP/drho ~ 10^6) so
    // fine resolution is needed near rho_max.

    double rho_lo_scan = 1.0;
    double rho_hi_scan = eos.rho_max * 0.98;

    // Two-pass scan: 501 coarse points + 501 points near the liquid region
    std::vector<double> scan_densities;

    // Pass 1: uniform coarse scan
    constexpr int N_COARSE = 500;
    for (int i = 0; i <= N_COARSE; ++i) {
        scan_densities.push_back(rho_lo_scan + (rho_hi_scan - rho_lo_scan) * i / N_COARSE);
    }

    // Pass 2: fine scan in the liquid region (top 20% of density range)
    double rho_liq_start = eos.rho_max * 0.5;
    constexpr int N_FINE = 500;
    for (int i = 0; i <= N_FINE; ++i) {
        scan_densities.push_back(rho_liq_start + (rho_hi_scan - rho_liq_start) * i / N_FINE);
    }

    // Sort and deduplicate
    std::sort(scan_densities.begin(), scan_densities.end());
    scan_densities.erase(std::unique(scan_densities.begin(), scan_densities.end(),
        [](double a, double b) { return std::abs(a - b) < 0.1; }), scan_densities.end());

    std::vector<double> scan_roots;

    double prev_f = eos.P(scan_densities[0], 1.0) - P_target;
    result.total_evals++;

    for (size_t i = 1; i < scan_densities.size(); ++i) {
        double rho = scan_densities[i];
        double f = eos.P(rho, 1.0) - P_target;
        result.total_evals++;

        if (prev_f * f < 0) {
            // Sign change -> bracket a root with bisection
            double lo = scan_densities[i - 1];
            double hi = rho;
            double f_lo = prev_f;

            for (int j = 0; j < 50; ++j) {
                double mid = 0.5 * (lo + hi);
                double f_mid = eos.P(mid, 1.0) - P_target;
                result.total_evals++;

                if (std::abs(f_mid) < 1.0) {
                    // Refine with Newton
                    double rho_r = mid;
                    for (int k = 0; k < 10; ++k) {
                        double fv = eos.P(rho_r, 1.0) - P_target;
                        double dfv = eos.dPdrho(rho_r, 1.0);
                        result.total_evals += 2;
                        if (std::abs(fv) < 0.01) break;
                        if (std::abs(dfv) < 1e-10) break;
                        rho_r -= fv / dfv;
                    }
                    scan_roots.push_back(rho_r);
                    break;
                }

                if (f_mid * f_lo < 0)
                    hi = mid;
                else {
                    lo = mid;
                    f_lo = f_mid;
                }
            }
        }

        prev_f = f;
    }

    // Also try to find the liquid root via bisection in the dense region.
    // At very low T, the liquid root is at eta close to close-packing where
    // P(rho) is extremely steep. Bisection is more reliable than Newton here.
    //
    // Strategy: scan from rho_max*0.5 to rho_max*0.99 at fine resolution,
    // looking specifically on the LIQUID branch (after the 2nd spinodal).
    {
        // First, find the upper spinodal (dP/drho = 0, the one at higher rho)
        // by scanning dP/drho from high to low density
        double rho_sp_upper = -1;
        {
            double rho_prev_sp = eos.rho_max * 0.98;
            double dp_prev_sp = eos.dPdrho(rho_prev_sp, 1.0);
            result.total_evals++;
            for (int j = 0; j < 200; ++j) {
                double rho_sp = eos.rho_max * (0.98 - 0.6 * (j + 1) / 200.0);
                double dp_sp = eos.dPdrho(rho_sp, 1.0);
                result.total_evals++;
                if (dp_prev_sp > 0 && dp_sp < 0) {
                    // Found upper spinodal: dP/drho goes from + to -
                    rho_sp_upper = 0.5 * (rho_prev_sp + rho_sp);
                    break;
                }
                rho_prev_sp = rho_sp;
                dp_prev_sp = dp_sp;
            }
        }

        // If upper spinodal found, search for liquid root between
        // upper spinodal and rho_max*0.99 using bisection
        if (rho_sp_upper > 0) {
            double rho_liq_lo = rho_sp_upper;
            double rho_liq_hi = eos.rho_max * 0.99;

            double f_lo = eos.P(rho_liq_lo, 1.0) - P_target;
            double f_hi = eos.P(rho_liq_hi, 1.0) - P_target;
            result.total_evals += 2;

            if (f_lo * f_hi < 0) {
                // Bracket exists: bisect
                for (int j = 0; j < 60; ++j) {
                    double mid = 0.5 * (rho_liq_lo + rho_liq_hi);
                    double f_mid = eos.P(mid, 1.0) - P_target;
                    result.total_evals++;
                    if (std::abs(f_mid) < 0.1) {
                        // Refine with Newton
                        double rho_r = mid;
                        for (int k = 0; k < 20; ++k) {
                            double fv = eos.P(rho_r, 1.0) - P_target;
                            double dfv = eos.dPdrho(rho_r, 1.0);
                            result.total_evals += 2;
                            if (std::abs(fv) < 0.01) break;
                            if (std::abs(dfv) < 1e-10) break;
                            double step_r = fv / dfv;
                            if (std::abs(step_r) > 0.1 * rho_r)
                                step_r = 0.1 * rho_r * (step_r > 0 ? 1 : -1);
                            rho_r -= step_r;
                        }
                        bool dup = false;
                        for (auto r : scan_roots) {
                            if (std::abs(r - rho_r) / std::max(r, 1.0) < 0.02) { dup = true; break; }
                        }
                        if (!dup && rho_r > 0) scan_roots.push_back(rho_r);
                        break;
                    }
                    if (f_mid * f_lo < 0)
                        rho_liq_hi = mid;
                    else {
                        rho_liq_lo = mid;
                        f_lo = f_mid;
                    }
                }
            }
        }
    }

    // Merge scan_roots with continuation roots
    for (auto rho_r : scan_roots) {
        bool duplicate = false;
        for (auto& r : result.roots) {
            if (std::abs(r - rho_r) / std::max(r, 1.0) < 0.01) {
                duplicate = true;
                break;
            }
        }
        if (!duplicate && rho_r > 0) {
            double dPdr = eos.dPdrho(rho_r, 1.0);
            result.total_evals++;
            if (dPdr > 0) {
                result.roots.push_back(rho_r);
                result.stable.push_back(true);
            }
        }
    }

    // Sort roots by density
    if (result.roots.size() > 1) {
        // Simple sort by density (with stability tag)
        std::vector<std::pair<double, bool>> paired;
        for (size_t i = 0; i < result.roots.size(); ++i) {
            paired.push_back({result.roots[i], result.stable[i]});
        }
        std::sort(paired.begin(), paired.end());
        for (size_t i = 0; i < paired.size(); ++i) {
            result.roots[i] = paired[i].first;
            result.stable[i] = paired[i].second;
        }
    }

    result.success = !result.roots.empty();

    return result;
}

// ============================================================
// Three-Level Fallback Solver
// ============================================================

struct FallbackResult {
    std::vector<double> roots;
    std::vector<bool>   stable;
    int    level_used  = -1;
    int    total_evals = 0;
    bool   success     = false;
};

inline FallbackResult fallback_solve(
    const EoSHomotopyInterface& eos,
    double P_target,
    double rho_hint,
    const HomotopyConfig& hcfg = {})
{
    FallbackResult fr;
    constexpr double TOL = 1.0;  // Pa

    auto f = [&](double rho) { return eos.P(rho, 1.0) - P_target; };
    auto df = [&](double rho) { return eos.dPdrho(rho, 1.0); };

    if (rho_hint > 0) {
        double rho = rho_hint;
        double dPdr = df(rho);
        fr.total_evals++;
        if (dPdr > 0) {
            double fval = f(rho);
            fr.total_evals++;
            if (std::abs(fval) < TOL) {
                fr.roots.push_back(rho);
                fr.stable.push_back(true);
                fr.level_used = 0;
                fr.success = true;
                return fr;
            }
            rho -= fval / dPdr;
            if (rho > 0 && rho < eos.rho_max) {
                fval = f(rho);
                dPdr = df(rho);
                fr.total_evals += 2;
                if (std::abs(fval) < TOL && dPdr > 0) {
                    fr.roots.push_back(rho);
                    fr.stable.push_back(true);
                    fr.level_used = 0;
                    fr.success = true;
                    return fr;
                }
            }
        }
    }

    constexpr double R_GAS = 8.314462;
    double rho_ideal = P_target / (R_GAS * eos.T);
    const double rho_lo_domain = std::max(eos.rho_min, 1e-3);
    const double rho_hi_domain = eos.rho_max * 0.98;

    // Level 1: safeguarded Newton from a gas-like and a dense-fluid start.
    // A Newton step is limited to half the current density and accepted by an
    // Armijo residual test.  If a sign bracket is available, bisection remains
    // available when Newton is singular or the line search is rejected.
    for (double rho0 : {rho_ideal, eos.rho_max * 0.7}) {
        double rho = std::max(rho_lo_domain, std::min(rho0, rho_hi_domain));

        double bracket_lo = rho_lo_domain;
        double bracket_hi = rho_hi_domain;
        double f_lo = f(bracket_lo);
        double f_hi = f(bracket_hi);
        fr.total_evals += 2;
        bool have_bracket = (f_lo == 0.0 || f_hi == 0.0 || f_lo * f_hi < 0.0);

        for (int i = 0; i < 50; ++i) {
            double fval = f(rho);
            double dfval = df(rho);
            fr.total_evals += 2;

            if (std::abs(fval) < TOL && dfval > 0) {
                fr.roots.push_back(rho);
                fr.stable.push_back(true);
                fr.level_used = 1;
                fr.success = true;
                break;
            }

            if (have_bracket) {
                if (f_lo * fval <= 0.0) {
                    bracket_hi = rho;
                    f_hi = fval;
                } else {
                    bracket_lo = rho;
                    f_lo = fval;
                }
            }

            bool accepted = false;
            double rho_next = rho;
            if (std::abs(dfval) >= 1e-10 * std::max(std::abs(fval), 1.0)) {
                double step = fval / dfval;
                double step_limit = 0.5 * std::max(rho, rho_lo_domain);
                step = std::max(-step_limit, std::min(step, step_limit));

                double alpha = 1.0;
                for (int backtrack = 0; backtrack <= 10; ++backtrack) {
                    double candidate = rho - alpha * step;
                    if (candidate > rho_lo_domain && candidate < rho_hi_domain) {
                        double f_candidate = f(candidate);
                        fr.total_evals++;
                        if (std::abs(f_candidate) <=
                            (1.0 - 1e-4 * alpha) * std::abs(fval)) {
                            if (!have_bracket && fval * f_candidate < 0.0) {
                                have_bracket = true;
                                if (rho < candidate) {
                                    bracket_lo = rho;
                                    f_lo = fval;
                                    bracket_hi = candidate;
                                    f_hi = f_candidate;
                                } else {
                                    bracket_lo = candidate;
                                    f_lo = f_candidate;
                                    bracket_hi = rho;
                                    f_hi = fval;
                                }
                            }
                            rho_next = candidate;
                            accepted = true;
                            break;
                        }
                    }
                    alpha *= 0.5;
                }
            }

            if (!accepted && have_bracket) {
                rho_next = 0.5 * (bracket_lo + bracket_hi);
                accepted = true;
            }
            if (!accepted || rho_next == rho) break;
            rho = rho_next;
        }
        if (fr.success) break;
    }

    if (fr.success) return fr;

    double rbar_anchor;
    {
        double rho = std::max(rho_lo_domain, std::min(rho_ideal, rho_hi_domain));
        for (int i = 0; i < 20; ++i) {
            double fval = eos.P(rho, 0.0) - P_target;
            double dfval = eos.dPdrho(rho, 0.0);
            fr.total_evals += 2;
            if (std::abs(fval) < TOL * 0.01) break;
            if (std::abs(dfval) < 1e-30) break;
            rho -= fval / dfval;
            rho = std::max(1e-3, std::min(rho, eos.rho_max * 0.99));
        }
        rbar_anchor = rho / eos.rho_ref;
    }

    auto hr = continuation_multi_branch(eos, P_target, rbar_anchor, hcfg);
    fr.total_evals += hr.total_evals;

    if (hr.success && !hr.roots.empty()) {
        fr.roots = hr.roots;
        fr.stable = hr.stable;
        fr.level_used = 2;
        fr.success = true;
    }

    return fr;
}

// ============================================================
// Mathematical (Linear Interpolation) Homotopy Interface
//
// Interpolates between ideal gas (λ=0) and full PC-SAFT (λ=1):
//   P_math(ρ, λ) = (1-λ)·ρ·R·T + λ·P_PCSAFT(ρ)
//   ∂P_math/∂ρ   = (1-λ)·R·T + λ·∂P_PCSAFT/∂ρ
//   ∂P_math/∂λ   = P_PCSAFT(ρ) - ρ·R·T
//
// At λ=0: P = ρ·R·T (ideal gas, unique root ρ = P_target/(R·T))
// At λ=1: P = P_PCSAFT(ρ) (full PC-SAFT)
//
// This is the "textbook" homotopy with NO physical embedding;
// used as ablation baseline against the physical dispersion-strength
// homotopy.
// ============================================================

inline EoSHomotopyInterface make_math_homotopy_interface(
    std::function<double(double rho)> P_pcsaft,      // P(ρ) at full PC-SAFT (λ_disp=1)
    std::function<double(double rho)> dPdrho_pcsaft,  // ∂P/∂ρ at full PC-SAFT
    double rho_min, double rho_max, double rho_ref, double T)
{
    constexpr double R_GAS = 8.314462;

    EoSHomotopyInterface eos;
    eos.T       = T;
    eos.rho_min = rho_min;
    eos.rho_max = rho_max;
    eos.rho_ref = rho_ref;

    // P(ρ, λ) = (1-λ)·ρ·R·T + λ·P_PCSAFT(ρ)
    eos.P = [P_pcsaft, T, R_GAS](double rho, double lam) -> double {
        if (rho <= 0) return 0;
        double P_ideal = rho * R_GAS * T;
        double P_full  = P_pcsaft(rho);
        return (1.0 - lam) * P_ideal + lam * P_full;
    };

    // ∂P/∂ρ = (1-λ)·R·T + λ·∂P_PCSAFT/∂ρ
    eos.dPdrho = [dPdrho_pcsaft, T, R_GAS](double rho, double lam) -> double {
        if (rho <= 0) return R_GAS * T;
        double dPdr_ideal = R_GAS * T;
        double dPdr_full  = dPdrho_pcsaft(rho);
        return (1.0 - lam) * dPdr_ideal + lam * dPdr_full;
    };

    // ∂P/∂λ = P_PCSAFT(ρ) - ρ·R·T
    eos.dPdlam = [P_pcsaft, T, R_GAS](double rho, double lam) -> double {
        (void)lam;
        if (rho <= 0) return 0;
        return P_pcsaft(rho) - rho * R_GAS * T;
    };

    return eos;
}

} // namespace homotopy
