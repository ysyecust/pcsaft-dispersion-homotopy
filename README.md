# PC-SAFT density root finding by dispersion-strength homotopy

Reference implementation for *A PC-SAFT Density Root-Finding Method Based on
Dispersion-Strength Homotopy and Pseudo-Arclength Continuation*.

The solver embeds the PC-SAFT density equation in its own hard-chain /
dispersion split, traces the anchor-connected solution curve with adaptive
pseudo-arclength continuation, and returns every intersection with the full
model as a density root, classified by `sign(dP/drho)`. It uses no density
grid, no stationary-point partition, and no multi-start rescue.

## Build

Requires a C++17 compiler and CMake 3.16+. No external dependencies.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

## 1. Smoke test — seconds

Solves one catalogue state and prints its classified root set beside roots
obtained independently by a stationary-point hierarchy.

```bash
./build/single_state_example              # a three-root state
./build/single_state_example --index 42   # any other catalogue state
```

Expected output ends with `returned 3 roots, reference 3 roots -> counts agree`.

## 2. Unit tests — under a minute

```bash
ctest --test-dir build --output-on-failure
```

`test_external_baselines` checks the fixed-point embedding algebraically before
it is used: the `lambda = 1` endpoint must reproduce the target model exactly,
the `lambda = 0` endpoint must have the anchor as its only root, and both
partial derivatives must agree with central differences.

## 3. Method comparison — reproduces Table 1 and Figure 3

Every method in Table 1 is evaluated on the same frozen catalogue against the
same independently verified reference roots.

```bash
# reduced catalogue, a few minutes
./build/external_baseline_benchmark --reference REFERENCE_ROOTS.csv \
    --output out_small --catalog validation --limit-per-group 250

# full 100,000-state catalogue as reported in the paper
./build/external_baseline_benchmark --reference REFERENCE_ROOTS.csv \
    --output out_full --catalog validation
```

`REFERENCE_ROOTS.csv` is `e2_root_results.csv` from the data archive. The full
run takes roughly two hours on one core of an Apple M4 Max; the reduced run is
representative to within a few tenths of a percent on every completeness
figure. Outputs are `cr_method_summary.csv` (per method),
`cr_class_summary.csv` (per method and target class) and
`cr_state_method_results.csv` (per state).

## 4. Structural audit of the two embeddings — about a minute

Samples both solution graphs and counts the poles of each, and applies the
starting-point criterion of Aslam and Sunol (2006) over the admissible
interval.

```bash
./build/embedding_pole_audit --output out_poles --limit-per-group 150
```

## Layout

| Path | Contents |
|---|---|
| `include/complete_homotopy_curve.hpp` | the solver: predictor–corrector, event detection, fold handling, termination |
| `include/fixed_point_homotopy.hpp` | global fixed-point homotopy baseline and its published starting-point criterion |
| `include/deflation_root_methods.hpp` | find-and-hide by deflation |
| `include/direct_root_methods.hpp` | Newton, uniform-scan and stationary-partition baselines |
| `include/reference_root_isolator.hpp` | independent stationary-point hierarchy used as reference |
| `include/pcsaft_*.hpp` | PC-SAFT pressure, derivatives and mixture combining rules |
| `include/*_catalog.hpp`, `src/*_catalog.cpp` | generators for the frozen state catalogues |

## Data

The frozen experiment outputs, model parameters and reference roots are
archived separately; see the Data availability statement of the article.

## Licence

MIT, see `LICENSE`.
