#pragma once
// ============================================================
// pcsaft_accurate.hpp — Production-grade PC-SAFT (pure component)
// Faithfully reimplemented from in-house simulator's pc_saft_item_calculator.cpp
//
// Key differences from the simplified pcsaft.hpp:
//  - Analytical Helmholtz derivatives (not finite-difference)
//  - BMCSL hard-sphere (not Carnahan-Starling approximation)
//  - Exact moment-based chain rule for ∂a/∂ρ
//  - Correct unit conversions (Å → m via 1e-30 factor)
// ============================================================

#include <cmath>
#include <array>
#include <algorithm>
#include <numeric>

namespace pcsaft_acc {

constexpr double PI  = 3.14159265358979323846;
constexpr double NA  = 6.02214199e23;
constexpr double R   = 8.314472;             // J/(mol·K) — in-house simulator value
constexpr double A3_TO_M3 = 1e-30;          // Å³ → m³

// ---- Gross-Sadowski universal constants (Table 2) ----

constexpr double A_COEFF[7][3] = {
    { 0.910563145,  -0.308401692, -0.090614836},
    { 0.636128145,   0.186053116,  0.452784281},
    { 2.686134789,  -2.503004726,  0.596270073},
    {-26.547362491,  21.419793600, -1.724182913},
    { 97.759208784, -65.255885330, -4.130211253},
    {-159.591540865,  83.318680481, 13.776631870},
    { 91.297774084, -33.746922930, -8.672847037},
};

constexpr double B_COEFF[7][3] = {
    { 0.724094694,  -0.575549808,  0.097688312},
    { 2.238279186,   0.699509552, -0.255757498},
    {-4.002584949,   3.892567339, -9.155856153},
    {-21.003576815, -17.215471648, 20.642075974},
    { 26.855641363, 192.672264500,-38.804430052},
    {206.551338100,-161.826461650, 93.626774077},
    {-355.602356120,-165.207693460,-29.666905585},
};

// ---- Component parameters ----

struct Component {
    const char* name;
    double m;       // segment number
    double sigma;   // segment diameter [Å]
    double eps_k;   // ε/k [K]
};

constexpr std::array<Component, 5> COMPONENTS = {{
    {"methane",  1.0000, 3.7039, 150.03},
    {"ethane",   1.6069, 3.5206, 191.42},
    {"propane",  2.0020, 3.6184, 208.11},
    {"n-butane", 2.3316, 3.7086, 222.88},
    {"CO2",      2.0729, 2.7852, 169.21},
}};

// ---- Helper: dispersion coefficients ----

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

// ---- Core PC-SAFT evaluation ----

struct PCSAFTResult {
    double P;       // pressure [Pa]
    double dPdrho;  // ∂P/∂ρ [Pa·m³/mol]
    double Z;       // compressibility factor
};

inline PCSAFTResult evaluate(const Component& c, double rho, double T,
                              double lambda_disp = 1.0) {
    // lambda_disp scales the dispersion contribution for homotopy
    double m = c.m;

    // Temperature-dependent segment diameter
    double d = c.sigma * (1.0 - 0.12 * std::exp(-3.0 * c.eps_k / T));
    double d2 = d * d;
    double d3 = d2 * d;

    // Moments: ζ_k = (π/6) * NA * ρ * m * d^k  (pure component)
    double prefactor = PI / 6.0 * NA * A3_TO_M3;
    double zeta0 = prefactor * rho * m;            // d^0 = 1
    double zeta1 = prefactor * rho * m * d;
    double zeta2 = prefactor * rho * m * d2;
    double zeta3 = prefactor * rho * m * d3;       // = η (packing fraction)
    double eta = zeta3;

    // ∂ζ_k/∂ρ (constant T)
    double dzeta0 = prefactor * m;
    double dzeta1 = prefactor * m * d;
    double dzeta2 = prefactor * m * d2;
    double dzeta3 = prefactor * m * d3;

    if (eta >= 0.74 || eta < 0.0) {
        return {1e30, 1e30, 1e30};
    }

    double om1 = 1.0 - eta;
    double om12 = om1 * om1;
    double om13 = om12 * om1;

    // ---- Hard-sphere Helmholtz: a^hs/RT (BMCSL) ----
    double z2_sq = zeta2 * zeta2;
    double z2_cu = z2_sq * zeta2;
    double z3_sq = zeta3 * zeta3;
    double ln_om1 = std::log(om1);

    double a_hs;
    if (zeta0 > 1e-30 && zeta3 > 1e-30) {
        double t1 = 3.0 * zeta1 * zeta2 / om1;
        double t2 = z2_cu / (zeta3 * om12);
        double t3 = (z2_cu / z3_sq - zeta0) * ln_om1;
        a_hs = (t1 + t2 + t3) / zeta0;
    } else {
        a_hs = 0.0;
    }

    // ∂a^hs/∂ρ via chain rule: ∂a^hs/∂ρ = Σ_k (∂a^hs/∂ζ_k)(∂ζ_k/∂ρ)
    // ∂a^hs/∂ζ_0:
    double da_dz0 = 0;
    if (zeta0 > 1e-30) {
        double bracket = 3.0*zeta1*zeta2/om1 + z2_cu/(zeta3*om12) +
                         (z2_cu/z3_sq - zeta0)*ln_om1;
        da_dz0 = -bracket/(zeta0*zeta0) - ln_om1/zeta0;
    }
    // ∂a^hs/∂ζ_1:
    double da_dz1 = (zeta0 > 1e-30) ? (3.0*zeta2/om1)/zeta0 : 0;
    // ∂a^hs/∂ζ_2:
    double da_dz2 = 0;
    if (zeta0 > 1e-30) {
        da_dz2 = (3.0*zeta1/om1 + 3.0*z2_sq/(zeta3*om12) +
                  3.0*z2_sq/z3_sq*ln_om1) / zeta0;
    }
    // ∂a^hs/∂ζ_3:
    double da_dz3 = 0;
    if (zeta0 > 1e-30) {
        double om14 = om13 * om1;
        da_dz3 = (3.0*zeta1*zeta2/om12
                  + z2_cu*(-1.0/(z3_sq*om12) + 2.0/(zeta3*om13))
                  - 2.0*z2_cu/(z3_sq*zeta3)*ln_om1
                  - (z2_cu/z3_sq - zeta0)/om1) / zeta0;
    }

    double da_hs_drho = da_dz0*dzeta0 + da_dz1*dzeta1 +
                        da_dz2*dzeta2 + da_dz3*dzeta3;

    // ---- Chain contribution: a^chain = -(m-1) ln(g_hs) ----
    // g_hs(σ) for pure component: BMCSL at contact
    // g = 1/(1-η) + 3d·ζ₂/(1-η)² + 2d²·ζ₂²/(1-η)³
    // For pure: d_ij = d/2 (harmonic mean = d*d/(d+d) = d/2)
    double dij = d / 2.0;
    double g_hs = 1.0/om1 + 3.0*dij*zeta2/om12 + 2.0*dij*dij*z2_sq/om13;

    double a_chain = -(m - 1.0) * std::log(g_hs);

    // ∂g_hs/∂ρ
    double dg_dz2 = 3.0*dij/om12 + 4.0*dij*dij*zeta2/om13;
    double dg_dz3 = 1.0/om12 + 6.0*dij*zeta2/om13 + 6.0*dij*dij*z2_sq/(om13*om1);
    double dg_drho = dg_dz2 * dzeta2 + dg_dz3 * dzeta3;

    double da_chain_drho = -(m - 1.0) * dg_drho / g_hs;

    // ---- Dispersion contribution ----
    double I1 = 0, dI1 = 0, I2 = 0, dI2 = 0;
    double eta_pow = 1.0;
    for (int l = 0; l < 7; ++l) {
        I1 += a_coeff(l, m) * eta_pow;
        I2 += b_coeff(l, m) * eta_pow;
        if (l > 0) {
            dI1 += l * a_coeff(l, m) * eta_pow / eta;
            dI2 += l * b_coeff(l, m) * eta_pow / eta;
        }
        eta_pow *= eta;
    }

    // Mixed interaction parameters (pure component simplification)
    // m²εσ³ = m² (ε/kT) σ³  (ε^1 power)
    double eps_over_kT = c.eps_k / T;
    double sigma3 = c.sigma * c.sigma * c.sigma;
    double m2e1s3 = m * m * eps_over_kT * sigma3;
    double m2e2s3 = m * m * eps_over_kT * eps_over_kT * sigma3;

    // Apply lambda_disp scaling (homotopy)
    m2e1s3 *= lambda_disp;
    m2e2s3 *= lambda_disp;

    // C₁ factor
    double eta2 = eta * eta, eta3 = eta2 * eta, eta4 = eta3 * eta;
    double om14 = om13 * om1;
    double two_m_eta = 2.0 - eta;
    double denom_sq = om1 * two_m_eta;
    denom_sq *= denom_sq;

    double C1_inv = 1.0 + m*(8.0*eta - 2.0*eta2)/om14
                   + (1.0-m)*(20.0*eta - 27.0*eta2 + 12.0*eta3 - 2.0*eta4)/denom_sq;
    double C1 = 1.0 / C1_inv;

    // ∂C₁/∂η
    double om15 = om14 * om1;
    double denom_cu = om1*two_m_eta; denom_cu = denom_cu*denom_cu*(om1*two_m_eta);
    double dC1_num1 = m * (-4.0*eta2 + 20.0*eta + 8.0) / om15;
    double dC1_num2 = (1.0-m) * (2.0*eta3 + 12.0*eta2 - 48.0*eta + 40.0) / denom_cu;
    double dC1_deta = -C1*C1 * (dC1_num1 + dC1_num2);

    // a^disp/RT = -π ρ NA [2 I₁ m²ε¹σ³ + m C₁ I₂ m²ε²σ³] × Å³→m³
    double a_disp = -PI * rho * NA * A3_TO_M3 *
                    (2.0 * I1 * m2e1s3 + m * C1 * I2 * m2e2s3);

    // ∂a^disp/∂ρ
    double da_disp_drho = -PI * NA * A3_TO_M3 * (
        // d/dρ [ρ · 2I₁ · m²ε¹σ³]  = 2m²ε¹σ³ (I₁ + ρ·dI₁/dρ)
        2.0 * m2e1s3 * (I1 + rho * dI1 * dzeta3)
        // d/dρ [ρ · m · C₁ · I₂ · m²ε²σ³]
        + m * m2e2s3 * (C1*I2 + rho*(dC1_deta*dzeta3*I2 + C1*dI2*dzeta3))
    );

    // ---- Total ----
    // a^res/RT = m·a^hs + a^chain + a^disp
    double a_res = m * a_hs + a_chain + a_disp;
    double da_res_drho = m * da_hs_drho + da_chain_drho + da_disp_drho;

    // Z = 1 + ρ · ∂a^res/∂ρ
    double Z = 1.0 + rho * da_res_drho;

    // P = ρ R T Z
    double P = rho * R * T * Z;

    // dP/dρ = R T (Z + ρ ∂Z/∂ρ) — use numerical for ∂Z/∂ρ
    // More precisely: dP/dρ = RT[Z + ρ(da_res_drho + ρ·d²a_res/dρ²)]
    // Use numerical differentiation for simplicity (same as in-house simulator)
    double h = std::max(rho * 1e-5, 1e-3);
    double rho_p = rho + h, rho_m = rho - h;

    // Recompute pressure at ρ±h for the central-difference derivative.
    // This is exactly what in-house simulator does (saft_model.cpp line 281-295)
    auto eval_full = [&](double r) -> double {
        // Recompute everything at density r
        double z3_ = prefactor * r * m * d3;
        double e_ = z3_;
        if (e_ >= 0.74 || e_ < 0) return 1e30;

        // Recompute all moments
        double z0_ = prefactor*r*m;
        double z1_ = prefactor*r*m*d;
        double z2_ = prefactor*r*m*d2;
        double o1_ = 1.0-e_;
        double o12_=o1_*o1_, o13_=o12_*o1_;

        // a_hs
        double zs2=z2_*z2_, zs3=zs2*z2_, es2=e_*e_;
        double ah=0;
        if(z0_>1e-30 && e_>1e-30){
            ah = (3.0*z1_*z2_/o1_ + zs3/(e_*o12_) + (zs3/es2-z0_)*std::log(o1_))/z0_;
        }

        // chain
        double dij_=d/2.0;
        double g_=1.0/o1_+3.0*dij_*z2_/o12_+2.0*dij_*dij_*zs2/o13_;
        double ac=-(m-1.0)*std::log(g_);

        // disp
        double I1_=0,I2_=0; double ep=1.0;
        for(int l=0;l<7;l++){I1_+=a_coeff(l,m)*ep;I2_+=b_coeff(l,m)*ep;ep*=e_;}
        double e2_=e_*e_,e3_=e2_*e_,e4_=e3_*e_;
        double o14_=o13_*o1_;
        double tm=2.0-e_; double ds=o1_*tm; ds*=ds;
        double C1i=1.0/(1.0+m*(8.0*e_-2.0*e2_)/o14_+(1.0-m)*(20.0*e_-27.0*e2_+12.0*e3_-2.0*e4_)/ds);
        double ad=-PI*r*NA*A3_TO_M3*(2.0*I1_*m2e1s3+m*C1i*I2_*m2e2s3);

        // Helmholtz → Z → P
        // Need ∂(a_res)/∂ρ at r. Use the analytical formula already computed...
        // Actually, simplest: just compute Z from a_res using the η-derivative route
        // Z = 1 + η · d(a_res)/dη where η = z3_
        // But a_res depends on η through all moments... this is circular.

        // Simplest correct approach: P = ρRT(1 + ρ · da_res/dρ)
        // We already have the analytical da_res_drho for the ORIGINAL rho.
        // For r, we need to recompute it. This is expensive but correct.

        // For the finite-difference dP/dρ, let's just compute P(ρ+h) and P(ρ-h)
        // using the existing full P computation path.
        // BUT we can't call evaluate() recursively without infinite dP/dρ loop.
        // Solution: compute P directly from Z = 1 + ρ·da/dρ without dP/dρ.

        // Quick analytical da/dρ at r:
        // (same chain-rule computation as above but at density r)
        double dz0=prefactor*m, dz1=prefactor*m*d, dz2_=prefactor*m*d2, dz3_=prefactor*m*d3;

        // ∂a_hs/∂ρ
        double da_hs=0;
        if(z0_>1e-30 && e_>1e-30){
            double ln_o1=std::log(o1_);
            double bkt=3.0*z1_*z2_/o1_+zs3/(e_*o12_)+(zs3/es2-z0_)*ln_o1;
            double daz0=-bkt/(z0_*z0_)-ln_o1/z0_;
            double daz1=3.0*z2_/(o1_*z0_);
            double daz2=(3.0*z1_/o1_+3.0*zs2/(e_*o12_)+3.0*zs2/es2*ln_o1)/z0_;
            double o14_2=o13_*o1_;
            double daz3=(3.0*z1_*z2_/o12_+zs3*(-1.0/(es2*o12_)+2.0/(e_*o13_))
                        -2.0*zs3/(es2*e_)*ln_o1-(zs3/es2-z0_)/o1_)/z0_;
            da_hs=daz0*dz0+daz1*dz1+daz2*dz2_+daz3*dz3_;
        }

        // ∂chain/∂ρ
        double dij2=dij_*dij_;
        double dgz2=3.0*dij_/o12_+4.0*dij2*z2_/o13_;
        double dgz3=1.0/o12_+6.0*dij_*z2_/o13_+6.0*dij2*zs2/(o13_*o1_);
        double dg_dr=dgz2*dz2_+dgz3*dz3_;
        double dac=-(m-1.0)*dg_dr/g_;

        // ∂disp/∂ρ
        double dI1_=0,dI2_=0; ep=1.0;
        for(int l=1;l<7;l++){
            ep=std::pow(e_,l-1);
            dI1_+=l*a_coeff(l,m)*ep;
            dI2_+=l*b_coeff(l,m)*ep;
        }
        double o15_=o14_*o1_;
        double dcu=o1_*tm; dcu=dcu*dcu*(o1_*tm);
        double dC1n1=m*(-4.0*e2_+20.0*e_+8.0)/o15_;
        double dC1n2=(1.0-m)*(2.0*e3_+12.0*e2_-48.0*e_+40.0)/dcu;
        double dC1=-C1i*C1i*(dC1n1+dC1n2);
        double dad=-PI*NA*A3_TO_M3*(
            2.0*m2e1s3*(I1_+r*dI1_*dz3_)+
            m*m2e2s3*(C1i*I2_+r*(dC1*dz3_*I2_+C1i*dI2_*dz3_)));

        double da_tot=m*da_hs+dac+dad;
        double Zr=1.0+r*da_tot;
        return r*R*T*Zr;
    };

    double P_plus = eval_full(rho_p);
    double P_minus = eval_full(rho_m);
    double dP = (P_plus - P_minus) / (2.0 * h);

    return {P, dP, Z};
}

// ---- Maximum density ----

inline double max_density(const Component& c, double T) {
    double d = c.sigma * (1.0 - 0.12 * std::exp(-3.0 * c.eps_k / T));
    double d3 = d * d * d;
    return 0.72 / (PI / 6.0 * NA * A3_TO_M3 * c.m * d3);
}

} // namespace pcsaft_acc
