"""
Diagnose TC7: why do active policies produce more collisions than no_op?
(non-high_risk_mode, 3 sats, 15 debris, collision_threshold=5.0 km, distance_threshold=100 km)
"""
import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
import numpy as np
from datetime import datetime
from env.ma_env import MultiAgentOrbitalEnv
from policies.policy_interface import NoOpPolicy, ThresholdRulePolicy

# Replicate TC7 exactly
env = MultiAgentOrbitalEnv(
    num_satellites=3,
    num_debris=15,
    collision_threshold_km=5.0,
    distance_threshold_km=100.0,
    high_risk_mode=False,
    dt=60.0,
    orbit_altitude_km=600.0,
    epoch_datetime=datetime(2020, 1, 1),
    initial_fuel_kg=1000.0,
    max_fuel_kg=1000.0,
    near_miss_distance_km=10.0,
    secondary_conjunction_risk_threshold=0.3,
)

print("=== TC7 initial state (high_risk_mode=False) ===")
obs = env.reset()
states = env._current_object_states()
ids = list(states.keys())
collision_threshold = env.detector.collision_threshold

# Check initial separations
close_sat_deb = []
for id1 in [k for k in ids if k.startswith("SAT_")]:
    for id2 in [k for k in ids if k.startswith("DEB_")]:
        d = float(np.linalg.norm(states[id1][:3] - states[id2][:3]))
        if d < collision_threshold:
            close_sat_deb.append((id1, id2, d))

print(f"Collision threshold: {collision_threshold} km")
print(f"SAT-DEB pairs within collision threshold at t=0: {len(close_sat_deb)}")
for id1, id2, d in sorted(close_sat_deb, key=lambda x: x[2])[:10]:
    print(f"  {id1} <-> {id2}: {d:.4f} km")

# Run no_op for 100 steps and track where collisions occur
print("\n=== no_op: stepping to find collision timing ===")
obs = env.reset()
noop_collisions_by_step = []
for step in range(100):
    actions = {aid: 0 for aid in obs}
    obs, _, dones, info = env.step(actions)
    if info["collisions_this_step"] > 0:
        noop_collisions_by_step.append((step + 1, info["collisions_this_step"], info["episode_collisions"]))
    if dones.get("__all__", False):
        print(f"  Episode ended at step {step+1} (total collisions={info['episode_collisions']})")
        break

print(f"  Steps with collisions: {noop_collisions_by_step}")
print(f"  Total no_op collisions: {env.episode_collisions}")

# Now run threshold_rule and compare
print("\n=== threshold_rule: stepping same env ===")
obs = env.reset()
policy = ThresholdRulePolicy(threshold_km=5.0, dv_action=1)
tr_collisions_by_step = []
maneuvers_applied = []
for step in range(100):
    actions = {}
    for aid, state in obs.items():
        if aid.startswith("SAT_"):
            actions[aid] = policy.select_action(state, aid)
        else:
            actions[aid] = 0
    obs, _, dones, info = env.step(actions)
    n_maneuvers = sum(1 for v in actions.values() if v != 0)
    if n_maneuvers > 0:
        maneuvers_applied.append((step + 1, n_maneuvers, dict(actions)))
    if info["collisions_this_step"] > 0:
        tr_collisions_by_step.append((step + 1, info["collisions_this_step"], info["episode_collisions"]))
    if dones.get("__all__", False):
        print(f"  Episode ended at step {step+1} (total collisions={info['episode_collisions']})")
        break

print(f"  Steps with collisions: {tr_collisions_by_step}")
print(f"  Maneuvers applied at steps: {[(s, n) for s, n, _ in maneuvers_applied[:10]]}")
print(f"  Total threshold_rule collisions: {env.episode_collisions}")
print()

# The issue: in non-high-risk mode, debris starts 20+ km away but
# maneuvers may MOVE the satellite INTO a debris cloud instead of away from it.
# Check: does action=1 (PROGRADE) push satellite toward any debris?
print("=== Final check: orbital geometry in non-high_risk_mode ===")
env2 = MultiAgentOrbitalEnv(
    num_satellites=3, num_debris=15,
    collision_threshold_km=5.0, distance_threshold_km=100.0,
    high_risk_mode=False, dt=60.0, orbit_altitude_km=600.0,
    epoch_datetime=datetime(2020, 1, 1), initial_fuel_kg=1000.0, max_fuel_kg=1000.0,
)
obs2 = env2.reset()
states2 = env2._current_object_states()
# How far is the closest SAT->DEB pair in non-high-risk mode?
min_sat_deb = float("inf")
for id1 in [k for k in states2 if k.startswith("SAT_")]:
    for id2 in [k for k in states2 if k.startswith("DEB_")]:
        d = float(np.linalg.norm(states2[id1][:3] - states2[id2][:3]))
        if d < min_sat_deb:
            min_sat_deb = d
print(f"Min SAT-DEB separation (non-high_risk): {min_sat_deb:.2f} km")
print(f"Debris non-high_risk SMA offset: 20+i*0.5 km => closest = 20 km")
print()
print("==> If min_sat_deb < 5 km at t=0 in TC7 then it's same Bug 1 in non-high_risk mode.")
print("==> If >= 5 km, collisions come from maneuver pushing SAT into debris during sim.")
