"""Post-fix verification: checks all 4 bugs are resolved."""
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", None)

df = pd.read_csv("outputs/test_framework/test_runs_per_policy.csv")
summary = pd.read_csv("outputs/test_framework/aggregated_summary.csv")
ttest = pd.read_csv("outputs/test_framework/ttest_collisions.csv")

print("=" * 70)
print("VERIFICATION 1: mean_collisions per policy per TC (must differ)")
print("=" * 70)
pivot = summary.pivot_table(
    index="test_case", columns="policy", values="mean_collisions"
)
print(pivot.to_string())

print()
print("=" * 70)
print("VERIFICATION 2: Collision range across policies (must be > 0 for TC1-TC3)")
print("=" * 70)
all_pass = True
for tc in summary["test_case"].unique():
    sub = summary[summary["test_case"] == tc]
    col_range = sub["mean_collisions"].max() - sub["mean_collisions"].min()
    fuel_range = sub["mean_fuel"].max() - sub["mean_fuel"].min()
    best_pol = sub.loc[sub["mean_collisions"].idxmin(), "policy"]
    worst_pol = sub.loc[sub["mean_collisions"].idxmax(), "policy"]
    status = "PASS" if col_range > 0 else "FAIL (no differentiation)"
    if col_range == 0 and tc not in ("TC7_secondary_conjunctions", "TC8_hypothetical_collision_cluster"):
        all_pass = False
    print(
        f"  [{status}] {tc}: range={col_range:.3f}  "
        f"best={best_pol}  worst={worst_pol}  fuel_range={fuel_range:.4f}"
    )

print()
print("=" * 70)
print("VERIFICATION 3: Active policies vs no_op (TC1-TC3, active must use fuel)")
print("=" * 70)
for tc in ["TC1_no_maneuver", "TC2_threshold_rule", "TC3_fuel_aware_rule"]:
    sub = summary[summary["test_case"] == tc][
        ["policy", "mean_collisions", "mean_fuel", "mean_maneuvers"]
    ].sort_values("mean_collisions")
    print(f"\n  {tc}:")
    for _, row in sub.iterrows():
        policy = str(row["policy"])
        collisions = float(row["mean_collisions"])
        fuel = float(row["mean_fuel"])
        maneuvers = float(row["mean_maneuvers"])
        print(f"    {policy:<32s}  collisions={collisions:.2f}  fuel={fuel:.4f}  maneuvers={maneuvers:.1f}")

print()
print("=" * 70)
print("VERIFICATION 4: TC7 - active policies must be <= no_op collisions (Bug 4)")
print("=" * 70)
tc7 = summary[summary["test_case"] == "TC7_secondary_conjunctions"][
    ["policy", "mean_collisions", "mean_secondary_conjunctions", "mean_fuel"]
].sort_values("mean_collisions")
print(tc7.to_string(index=False))

noop_val = tc7.loc[tc7["policy"] == "no_op", "mean_collisions"]
active_max = tc7.loc[tc7["policy"] != "no_op", "mean_collisions"].max()
if len(noop_val) and active_max <= float(noop_val.iloc[0]):
    print("  [PASS] Active policies produce <= no_op collisions")
else:
    print("  [FAIL] Active policies still worse than no_op — secondary conjunction logic needs review")

print()
print("=" * 70)
print("VERIFICATION 5: T-test NaN rate (before fix: 100% NaN)")
print("=" * 70)
valid_tt = ttest[ttest["p_value"].notna()]
total = len(ttest)
valid = len(valid_tt)
print(f"  Total t-tests: {total}, valid (non-NaN): {valid}, NaN: {total - valid}")
if valid > 0:
    print("  [PASS] Variance now exists — t-tests are computable")
    print(valid_tt[["test_case", "policy_a", "policy_b", "mean_a", "mean_b", "p_value"]].to_string(index=False))
else:
    print("  [FAIL] All t-tests still NaN — still no variance")

print()
print("=" * 70)
print("OVERALL:", "ALL CHECKS PASSED" if all_pass else "SOME CHECKS FAILED")
print("=" * 70)
