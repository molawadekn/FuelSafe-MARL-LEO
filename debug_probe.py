"""Debug probe: inspect initial separations + collision step behaviour."""
import sys, warnings, numpy as np
warnings.filterwarnings('ignore')
sys.path.insert(0, '.')
from datetime import datetime
from env.ma_env import MultiAgentOrbitalEnv

env = MultiAgentOrbitalEnv(
    num_satellites=3, num_debris=20,
    collision_threshold_km=5.0, distance_threshold_km=250.0,
    high_risk_mode=True, dt=60.0, orbit_altitude_km=600.0,
    epoch_datetime=datetime(2020, 1, 1),
    initial_fuel_kg=1000.0, max_fuel_kg=1000.0,
)
obs = env.reset()

states = env._current_object_states()
ids = list(states.keys())
min_dist = float('inf')
close_pairs = []
for i, id1 in enumerate(ids):
    for id2 in ids[i+1:]:
        d = float(np.linalg.norm(states[id1][:3] - states[id2][:3]))
        if d < 10.0:
            close_pairs.append((id1, id2, d))
        if d < min_dist:
            min_dist = d

print(f"Min separation at reset: {min_dist:.4f} km")
print(f"Number of pairs within 10 km: {len(close_pairs)}")
print("Closest pairs (sorted):")
close_pairs.sort(key=lambda x: x[2])
for id1, id2, d in close_pairs[:20]:
    print(f"  {id1} <-> {id2}: {d:.6f} km")
print()

# Check how many pairs are already within collision threshold at t=0
print(f"Collision threshold: {env.detector.collision_threshold} km")
already_colliding = [x for x in close_pairs if x[2] < env.detector.collision_threshold]
print(f"Pairs ALREADY within collision threshold at t=0: {len(already_colliding)}")
for id1, id2, d in already_colliding:
    print(f"  {id1} <-> {id2}: {d:.6f} km [COLLISION AT INIT]")
print()

# Step through no_op for 5 steps and track collisions
print("Stepping with no_op (all agents action=0):")
for step in range(10):
    actions = {aid: 0 for aid in list(obs.keys())}
    obs, rewards, dones, info = env.step(actions)
    print(f"  Step {step+1}: new_collisions={info['collisions_this_step']}, episode_total={info['episode_collisions']}, alerts={info['alerts_count']}")
    if info['episode_collisions'] >= 5:
        print("  [EARLY TERMINATION triggered: episode_collisions >= 5]")
        break
print()

# CRITICAL: Check if collisions happen BEFORE any maneuver is possible
# i.e., are they baked in by the initial orbital config?
print("Checking if episode_collisions caps at 5 (early termination):")
print(f"  _compute_dones early-exit threshold: episode_collisions >= 5")
print(f"  => With {len(already_colliding)} pairs already within {env.detector.collision_threshold} km at t=0,")
print(f"     early termination fires at step 1 and no policy can prevent initial collisions.")
