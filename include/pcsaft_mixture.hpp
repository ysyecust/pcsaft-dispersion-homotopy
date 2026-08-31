#pragma once
// ============================================================
// pcsaft_mixture.hpp — N-component PC-SAFT mixture
//
// Implements the full multi-component PC-SAFT EoS following
// Gross & Sadowski (2001) with Lorentz-Berthelot combining rules.
//
// Uses the SAME universal constants (A_COEFF, B_COEFF) from
// pcsaft_accurate.hpp.
//
// Key features:
//   - Arbitrary number of components
//   - Binary interaction parameters (kij)
//   - Dispersion-strength homotopy via lambda_disp parameter
//   - Numerical dP/dρ via central difference
// ============================================================

#include <cmath>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cassert>

// Import universal constants from the pure-component header
#include "pcsaft_accurate.hpp"

namespace pcsaft_mix {

using pcsaft_acc::PI;
using pcsaft_acc::NA;
using pcsaft_acc::R;
using pcsaft_acc::A3_TO_M3;
using pcsaft_acc::A_COEFF;
using pcsaft_acc::B_COEFF;

// ---- Dispersion coefficient helpers (same as pure, parameterized by m_bar) ----

inline double a_coeff(int l, double m_bar) {
    double m1 = (m_bar - 1.0) / m_bar;
    double m2 = m1 * (m_bar - 2.0) / m_bar;
    return A_COEFF[l][0] + m1 * A_COEFF[l][1] + m2 * A_COEFF[l][2];
}

inline double b_coeff(int l, double m_bar) {
    double m1 = (m_bar - 1.0) / m_bar;
    double m2 = m1 * (m_bar - 2.0) / m_bar;
    return B_COEFF[l][0] + m1 * B_COEFF[l][1] + m2 * B_COEFF[l][2];
}

// ---- Mixture parameters ----

struct MixtureParams {
    int nc;                                        // number of components
    std::vector<double> m;                         // segment numbers
    std::vector<double> sigma;                     // segment diameters [Angstrom]
    std::vector<double> eps_k;                     // epsilon/k [K]
    std::vector<std::vector<double>> kij;          // binary interaction parameters
};

struct MixResult {
    double P;       // pressure [Pa]
    double dPdrho;  // dP/dρ [Pa·m³/mol]
};

// ---- Core: evaluate mixture P at given (rho, T, x, lambda_disp) ----
// Returns pressure P only (no dP/dρ); used internally and for dP/dρ
// via central difference.

inline double evaluate_P(const MixtureParams& params,
                         const std::vector<double>& x,
                         double rho, double T,
                         double lambda_disp = 1.0) {
    const int nc = params.nc;
    if (rho <= 0) return 0.0;

    // Temperature-dependent segment diameters
    std::vector<double> d(nc);
    for (int i = 0; i < nc; ++i) {
        d[i] = params.sigma[i] * (1.0 - 0.12 * std::exp(-3.0 * params.eps_k[i] / T));
    }

    // Mean segment number: m_bar = sum(xi * mi)
    double m_bar = 0.0;
    for (int i = 0; i < nc; ++i) m_bar += x[i] * params.m[i];

    // Moments: zeta_k = (pi/6) * rho * NA * 1e-30 * sum(xi * mi * di^k)
    double prefactor = PI / 6.0 * NA * A3_TO_M3 * rho;
    double zeta0 = 0, zeta1 = 0, zeta2 = 0, zeta3 = 0;
    for (int i = 0; i < nc; ++i) {
        double di2 = d[i] * d[i];
        double di3 = di2 * d[i];
        zeta0 += x[i] * params.m[i];
        zeta1 += x[i] * params.m[i] * d[i];
        zeta2 += x[i] * params.m[i] * di2;
        zeta3 += x[i] * params.m[i] * di3;
    }
    zeta0 *= prefactor / rho;  // zeta0 = prefactor_base * sum
    // Actually recompute properly:
    zeta0 = PI / 6.0 * NA * A3_TO_M3 * rho * zeta0 / (PI / 6.0 * NA * A3_TO_M3 * rho);
    // Let me redo this cleanly:
    {
        double sum0 = 0, sum1 = 0, sum2 = 0, sum3 = 0;
        for (int i = 0; i < nc; ++i) {
            double di2 = d[i] * d[i];
            double di3 = di2 * d[i];
            sum0 += x[i] * params.m[i];
            sum1 += x[i] * params.m[i] * d[i];
            sum2 += x[i] * params.m[i] * di2;
            sum3 += x[i] * params.m[i] * di3;
        }
        double pf = PI / 6.0 * NA * A3_TO_M3 * rho;
        zeta0 = pf * sum0;
        zeta1 = pf * sum1;
        zeta2 = pf * sum2;
        zeta3 = pf * sum3;
    }

    double eta = zeta3;
    if (eta >= 0.74 || eta < 0.0) return 1e30;

    double om1 = 1.0 - eta;
    double om12 = om1 * om1;
    double om13 = om12 * om1;

    // ---- Hard-sphere Helmholtz: a^hs/RT (BMCSL) ----
    double z2_sq = zeta2 * zeta2;
    double z2_cu = z2_sq * zeta2;
    double z3_sq = zeta3 * zeta3;
    double ln_om1 = std::log(om1);

    double a_hs = 0.0;
    if (zeta0 > 1e-30 && zeta3 > 1e-30) {
        double t1 = 3.0 * zeta1 * zeta2 / om1;
        double t2 = z2_cu / (zeta3 * om12);
        double t3 = (z2_cu / z3_sq - zeta0) * ln_om1;
        a_hs = (t1 + t2 + t3) / zeta0;
    }

    // ---- Hard-chain: a^hc/RT = m_bar * a^hs - sum(xi*(mi-1)*ln(g_ii)) ----
    // g_ij (BMCSL at contact):
    //   sigma_tilde_ij = di * dj / (di + dj)
    //   g_ij = 1/(1-zeta3) + 3*sigma_tilde_ij*zeta2/(1-zeta3)^2
    //          + 2*sigma_tilde_ij^2*zeta2^2/(1-zeta3)^3

    double a_hc = m_bar * a_hs;
    for (int i = 0; i < nc; ++i) {
        double sigma_tilde_ii = d[i] * d[i] / (d[i] + d[i]);  // = d[i]/2
        double g_ii = 1.0 / om1
                    + 3.0 * sigma_tilde_ii * zeta2 / om12
                    + 2.0 * sigma_tilde_ii * sigma_tilde_ii * z2_sq / om13;
        a_hc -= x[i] * (params.m[i] - 1.0) * std::log(g_ii);
    }

    // ---- Dispersion contribution ----
    // Mixing rules for dispersion sums:
    //   S^(1) = sum_i sum_j xi xj mi mj (eps_ij/kT) sigma_ij^3
    //   S^(2) = sum_i sum_j xi xj mi mj (eps_ij/kT)^2 sigma_ij^3
    //
    // Combining rules (Lorentz-Berthelot):
    //   sigma_ij = (sigma_i + sigma_j) / 2
    //   eps_ij = sqrt(eps_i * eps_j) * (1 - kij)

    double S1 = 0.0, S2 = 0.0;
    for (int i = 0; i < nc; ++i) {
        for (int j = 0; j < nc; ++j) {
            double sigma_ij = 0.5 * (params.sigma[i] + params.sigma[j]);
            double eps_ij_k = std::sqrt(params.eps_k[i] * params.eps_k[j])
                            * (1.0 - params.kij[i][j]);
            double sigma_ij3 = sigma_ij * sigma_ij * sigma_ij;
            double ekT = eps_ij_k / T;
            S1 += x[i] * x[j] * params.m[i] * params.m[j] * ekT * sigma_ij3;
            S2 += x[i] * x[j] * params.m[i] * params.m[j] * ekT * ekT * sigma_ij3;
        }
    }

    // Apply homotopy scaling
    S1 *= lambda_disp;
    S2 *= lambda_disp;

    // Power series I1, I2
    double I1 = 0.0, I2 = 0.0;
    double eta_pow = 1.0;
    for (int l = 0; l < 7; ++l) {
        I1 += a_coeff(l, m_bar) * eta_pow;
        I2 += b_coeff(l, m_bar) * eta_pow;
        eta_pow *= eta;
    }

    // C1 factor
    double eta2 = eta * eta, eta3 = eta2 * eta, eta4 = eta3 * eta;
    double om14 = om13 * om1;
    double two_m_eta = 2.0 - eta;
    double denom_sq = om1 * two_m_eta;
    denom_sq *= denom_sq;

    double C1_inv = 1.0 + m_bar * (8.0 * eta - 2.0 * eta2) / om14
                  + (1.0 - m_bar) * (20.0 * eta - 27.0 * eta2 + 12.0 * eta3 - 2.0 * eta4) / denom_sq;
    double C1 = 1.0 / C1_inv;

    // a^disp/RT = -pi * rho * NA * 1e-30 * [2*I1*S1 + m_bar*C1*I2*S2]
    double a_disp = -PI * rho * NA * A3_TO_M3
                  * (2.0 * I1 * S1 + m_bar * C1 * I2 * S2);

    // ---- Total residual Helmholtz ----
    double a_res = a_hc + a_disp;

    // ---- Z and P via analytical ∂a^res/∂ρ ----
    // We compute ∂a^res/∂ρ analytically via the chain rule through moments.
    //
    // ∂ζ_k/∂ρ = ζ_k / ρ  (since ζ_k is linear in ρ)
    double dzeta0 = zeta0 / rho;
    double dzeta1 = zeta1 / rho;
    double dzeta2 = zeta2 / rho;
    double dzeta3 = zeta3 / rho;

    // ∂a^hs/∂ρ via chain rule: sum_k (∂a^hs/∂ζ_k)(∂ζ_k/∂ρ)
    double da_hs_drho = 0.0;
    if (zeta0 > 1e-30 && zeta3 > 1e-30) {
        double bracket = 3.0 * zeta1 * zeta2 / om1 + z2_cu / (zeta3 * om12)
                       + (z2_cu / z3_sq - zeta0) * ln_om1;
        double da_dz0 = -bracket / (zeta0 * zeta0) - ln_om1 / zeta0;
        double da_dz1 = (3.0 * zeta2 / om1) / zeta0;
        double da_dz2 = (3.0 * zeta1 / om1 + 3.0 * z2_sq / (zeta3 * om12)
                       + 3.0 * z2_sq / z3_sq * ln_om1) / zeta0;
        double om14_ = om13 * om1;
        double da_dz3 = (3.0 * zeta1 * zeta2 / om12
                       + z2_cu * (-1.0 / (z3_sq * om12) + 2.0 / (zeta3 * om13))
                       - 2.0 * z2_cu / (z3_sq * zeta3) * ln_om1
                       - (z2_cu / z3_sq - zeta0) / om1) / zeta0;
        (void)om14_;
        da_hs_drho = da_dz0 * dzeta0 + da_dz1 * dzeta1 + da_dz2 * dzeta2 + da_dz3 * dzeta3;
    }

    // ∂a^hc/∂ρ = m_bar * ∂a^hs/∂ρ - sum_i xi(mi-1) * (1/g_ii) * ∂g_ii/∂ρ
    double da_hc_drho = m_bar * da_hs_drho;
    for (int i = 0; i < nc; ++i) {
        double sigma_tilde_ii = d[i] / 2.0;
        double st2 = sigma_tilde_ii * sigma_tilde_ii;
        double g_ii = 1.0 / om1 + 3.0 * sigma_tilde_ii * zeta2 / om12
                    + 2.0 * st2 * z2_sq / om13;

        // ∂g_ii/∂ρ = (∂g_ii/∂ζ2)(∂ζ2/∂ρ) + (∂g_ii/∂ζ3)(∂ζ3/∂ρ)
        double dg_dz2 = 3.0 * sigma_tilde_ii / om12 + 4.0 * st2 * zeta2 / om13;
        double dg_dz3 = 1.0 / om12 + 6.0 * sigma_tilde_ii * zeta2 / om13
                      + 6.0 * st2 * z2_sq / (om13 * om1);
        double dg_drho = dg_dz2 * dzeta2 + dg_dz3 * dzeta3;

        da_hc_drho -= x[i] * (params.m[i] - 1.0) * dg_drho / g_ii;
    }

    // ∂a^disp/∂ρ
    // dI1/dη and dI2/dη
    double dI1 = 0.0, dI2 = 0.0;
    eta_pow = 1.0;
    for (int l = 1; l < 7; ++l) {
        double ep = std::pow(eta, l - 1);
        dI1 += l * a_coeff(l, m_bar) * ep;
        dI2 += l * b_coeff(l, m_bar) * ep;
    }

    // dC1/dη
    double om15 = om14 * om1;
    double denom_cu = om1 * two_m_eta;
    denom_cu = denom_cu * denom_cu * (om1 * two_m_eta);
    double dC1_num1 = m_bar * (-4.0 * eta2 + 20.0 * eta + 8.0) / om15;
    double dC1_num2 = (1.0 - m_bar) * (2.0 * eta3 + 12.0 * eta2 - 48.0 * eta + 40.0) / denom_cu;
    double dC1_deta = -C1 * C1 * (dC1_num1 + dC1_num2);

    double da_disp_drho = -PI * NA * A3_TO_M3 * (
        2.0 * S1 * (I1 + rho * dI1 * dzeta3)
      + m_bar * S2 * (C1 * I2 + rho * (dC1_deta * dzeta3 * I2 + C1 * dI2 * dzeta3))
    );

    // Total ∂a^res/∂ρ
    double da_res_drho = da_hc_drho + da_disp_drho;

    // Z = 1 + ρ * ∂a^res/∂ρ
    double Z = 1.0 + rho * da_res_drho;

    // P = ρ R T Z
    return rho * R * T * Z;
}

// ---- Full evaluation with numerical dP/dρ ----

inline MixResult evaluate_mixture(const MixtureParams& params,
                                  const std::vector<double>& x,
                                  double rho, double T,
                                  double lambda_disp = 1.0) {
    double P = evaluate_P(params, x, rho, T, lambda_disp);

    // dP/dρ via central difference (same approach as in-house simulator)
    double h = std::max(rho * 1e-5, 1e-3);
    double P_plus  = evaluate_P(params, x, rho + h, T, lambda_disp);
    double P_minus = evaluate_P(params, x, rho - h, T, lambda_disp);
    double dPdrho = (P_plus - P_minus) / (2.0 * h);

    return {P, dPdrho};
}

// ---- Maximum packing density for mixture ----

inline double max_density_mix(const MixtureParams& params,
                              const std::vector<double>& x, double T) {
    // eta_max ~ 0.74 => rho_max = 0.72 / (pi/6 * NA * 1e-30 * sum(xi*mi*di^3))
    double sum_d3 = 0.0;
    for (int i = 0; i < params.nc; ++i) {
        double di = params.sigma[i] * (1.0 - 0.12 * std::exp(-3.0 * params.eps_k[i] / T));
        sum_d3 += x[i] * params.m[i] * di * di * di;
    }
    return 0.72 / (PI / 6.0 * NA * A3_TO_M3 * sum_d3);
}

} // namespace pcsaft_mix
