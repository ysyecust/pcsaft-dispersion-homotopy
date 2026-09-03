"""Offline analysis of the tangent-state root dump written by
external_baseline_benchmark --only-tangent --dump-roots.

Re-implements the labelled stable-subset comparison of
full_solver_benchmark.hpp (same density-matching window) so that the number
of tangent states whose stable-root labels disagree with the reference can be
evaluated at any chi threshold, and reports the indicator magnitudes and
density differences of the disagreeing states of the proposed method."""
import csv, math, os, sys, collections, statistics
from pathlib import Path

ROOT = Path(os.environ.get("PCSAFT_REVIEW_ROOT", Path(__file__).resolve().parents[2]))
DUMP = ROOT / "data" / "experiment_cr_tangent_labels" / "tangent_roots.csv"
OUT = ROOT / "data" / "experiment_cr_tangent_labels"
REL_TOL = 2e-6
METHODS = ["dispersion_homotopy_pseudo_arclength", "stationary_partition_8192",
           "stationary_partition_512", "newton_six", "newton_six_safeguarded",
           "find_and_hide_deflation", "fixed_point_homotopy_criterion"]

roots = collections.defaultdict(list)          # (state, method) -> [root dict]
cls = {}
with DUMP.open() as fh:
    for r in csv.DictReader(fh):
        roots[(r["state_id"], r["method"])].append(
            {k: float(r[k]) for k in ("density", "chi", "residual", "derivative")})
        cls[r["state_id"]] = r["target_class"]
states = sorted(cls)

def unc(root):
    d = root["derivative"]
    if not math.isfinite(root["residual"]) or not math.isfinite(d) or abs(d) < 1e-30:
        return 0.0
    return abs(root["residual"] / d)

def same_root(a, b):
    rel = REL_TOL * max(1.0, abs(a["density"]), abs(b["density"]))
    win = 2.0 * (unc(a) + unc(b))
    return abs(a["density"] - b["density"]) <= max(rel, win)

def stable_subset_agrees(ref, ret, tau):
    rs = [r for r in ref if r["chi"] > tau]
    ts = [r for r in ret if r["chi"] > tau]
    used = [False] * len(rs)
    matched = 0
    for t in ts:
        best, bestd = None, math.inf
        for i, r in enumerate(rs):
            if used[i] or not same_root(r, t):
                continue
            d = abs(r["density"] - t["density"])
            if d < bestd:
                best, bestd = i, d
        if best is None:
            return False            # extra stable candidate
        used[best] = True
        matched += 1
    return matched == len(rs)       # no missing stable root

taus = [10 ** e for e in (-12, -11, -10, -9.5, -9, -8.5, -8, -7.5, -7, -6.5, -6, -5)]
table = {m: [] for m in METHODS}
for tau in taus:
    for m in METHODS:
        bad = sum(0 if stable_subset_agrees(roots[(s, "reference")], roots[(s, m)], tau) else 1
                  for s in states)
        table[m].append(bad)

with (OUT / "label_disagreement_vs_tau.csv").open("w") as fh:
    w = csv.writer(fh)
    w.writerow(["tau_chi"] + METHODS)
    for i, tau in enumerate(taus):
        w.writerow([f"{tau:.3g}"] + [table[m][i] for m in METHODS])
print("tangent states:", len(states))
print("tau      " + "  ".join(f"{m[:14]:>14s}" for m in METHODS))
for i, tau in enumerate(taus):
    print(f"{tau:8.1e} " + "  ".join(f"{table[m][i]:14d}" for m in METHODS))

# --- the disagreeing states of the proposed method at tau = 1e-9 ---
ours = "dispersion_homotopy_pseudo_arclength"
tau0 = 1e-9
rows = []
for s in states:
    ref, ret = roots[(s, "reference")], roots[(s, ours)]
    if stable_subset_agrees(ref, ret, tau0):
        continue
    # find the tangent root: the reference root with the smallest |chi|
    rt = min(ref, key=lambda r: abs(r["chi"]))
    cand = [t for t in ret if same_root(rt, t)]
    tt = min(cand, key=lambda t: abs(t["density"] - rt["density"])) if cand else None
    rows.append({"state_id": s, "class": cls[s], "chi_ref": rt["chi"],
                 "chi_ret": tt["chi"] if tt else float("nan"),
                 "rel_density_diff": abs(tt["density"] - rt["density"]) / rt["density"] if tt else float("nan"),
                 "residual_ret_Pa": tt["residual"] if tt else float("nan"),
                 "n_ref": len(ref), "n_ret": len(ret)})
with (OUT / "proposed_method_label_disagreements.csv").open("w") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print("\nproposed method, disagreeing tangent states at tau=1e-9:", len(rows))
print(" classes:", collections.Counter(r["class"] for r in rows))
print(" density matched in all:", all(math.isfinite(r["rel_density_diff"]) for r in rows),
      "| same root count:", sum(r["n_ref"] == r["n_ret"] for r in rows))
for k in ("chi_ref", "chi_ret", "rel_density_diff"):
    v = sorted(abs(r[k]) for r in rows if math.isfinite(r[k]))
    print(f" |{k}|: min={v[0]:.2e} median={statistics.median(v):.2e} max={v[-1]:.2e}")
print(" sign pattern (ref label, ret label):", collections.Counter(
    (("stable" if r["chi_ref"] > tau0 else "marginal" if r["chi_ref"] >= -tau0 else "unstable"),
     ("stable" if r["chi_ret"] > tau0 else "marginal" if r["chi_ret"] >= -tau0 else "unstable")) for r in rows))
# how do the reference tangent roots look on ALL 2000 states
allref = [min(roots[(s, "reference")], key=lambda r: abs(r["chi"]))["chi"] for s in states]
print(" reference tangent-root |chi| over 2000 states: median=%.2e max=%.2e" % (statistics.median(abs(x) for x in allref), max(abs(x) for x in allref)))
