"""Quick verification script for the realism layer integration."""
import sys
import numpy as np
from datetime import datetime

sys.path.insert(0, ".")

from sim.realism import RealismConfig, RealismLayer
from env.ma_env import MultiAgentOrbitalEnv
from sim.evaluation import compute_detailed_metrics

def test_realism_disabled():
    """Existing behavior: realism OFF should work exactly as before."""
    print("=" * 60)
    print("TEST 1: Realism DISABLED (backward compatibility)")
    print("=" * 60)
    env = MultiAgentOrbitalEnv(
        num_satellites=3,
        num_debris=5,
        collision_threshold_km=0.025,
        distance_threshold_km=50.0,
        high_risk_mode=True,
        realism_config=RealismConfig(enabled=False),
        epoch_datetime=datetime(2020, 1, 1),
    )
    obs = env.reset()
    print(f"  Reset OK. {len(obs)} agent observations.")
    
    total_collisions = 0
    for step in range(20):
        actions = {aid: 0 for aid in list(obs.keys())}
        obs, rewards, dones, info = env.step(actions)
        total_collisions = info["episode_collisions"]
    
    print(f"  20 steps completed. Collisions: {total_collisions}")
    print(f"  Realism enabled in info: {info.get('realism_enabled', 'N/A')}")
    assert info["realism_enabled"] == False, "Should report realism disabled"
    print("  PASSED\n")


def test_realism_enabled():
    """Realism ON: observations should be noisy, delays should buffer."""
    print("=" * 60)
    print("TEST 2: Realism ENABLED")
    print("=" * 60)
    config = RealismConfig(enabled=True)
    env = MultiAgentOrbitalEnv(
        num_satellites=3,
        num_debris=5,
        collision_threshold_km=0.025,
        distance_threshold_km=50.0,
        high_risk_mode=True,
        realism_config=config,
        epoch_datetime=datetime(2020, 1, 1),
    )
    obs = env.reset()
    print(f"  Reset OK. {len(obs)} agent observations.")
    
    total_collisions = 0
    for step in range(20):
        # Use action 1 (prograde) to test maneuver delay
        actions = {aid: 1 for aid in list(obs.keys())}
        obs, rewards, dones, info = env.step(actions)
        total_collisions = info["episode_collisions"]
    
    print(f"  20 steps completed. Collisions: {total_collisions}")
    print(f"  Realism enabled in info: {info.get('realism_enabled', 'N/A')}")
    print(f"  Min distance (m): {info.get('min_distance_m', 'N/A')}")
    print(f"  Fuel remaining: {info.get('fuel_remaining', 'N/A')}")
    assert info["realism_enabled"] == True, "Should report realism enabled"
    print("  PASSED\n")


def test_tc8_hardened():
    """TC8 scenario should use the new tighter thresholds."""
    print("=" * 60)
    print("TEST 3: TC8 scenario hardening")
    print("=" * 60)
    np.random.seed(42)
    env = MultiAgentOrbitalEnv(
        num_satellites=3,
        num_debris=15,
        collision_threshold_km=0.025,
        distance_threshold_km=50.0,
        realism_config=RealismConfig(enabled=False),
        epoch_datetime=datetime(2020, 1, 1),
    )
    scenario = env.generate_tc8_scenario()
    print(f"  TC8 cluster count: {len(scenario.get('cluster_offsets', []))}")
    print(f"  TC8 collision_threshold_km: {scenario.get('collision_threshold_km')}")
    
    miss_distances = [
        entry["conjunction_info"]["miss_distance"]
        for entry in scenario.get("cluster_offsets", [])
    ]
    if miss_distances:
        print(f"  Miss distance range: {min(miss_distances):.1f} - {max(miss_distances):.1f} m")
    
    assert scenario["collision_threshold_km"] == 0.005, f"Expected 0.005, got {scenario['collision_threshold_km']}"
    assert all(5.0 <= d <= 50.0 for d in miss_distances), f"Miss distances outside 5-50m range"
    print("  PASSED\n")


def test_conjunction_detector_threshold():
    """Conjunction detector default should now be 0.005 km."""
    print("=" * 60)
    print("TEST 4: ConjunctionDetector default threshold")
    print("=" * 60)
    from sim.conjunction_detector import ConjunctionDetector
    det = ConjunctionDetector()
    print(f"  Default collision_threshold: {det.collision_threshold} km ({det.collision_threshold * 1000:.0f} m)")
    assert det.collision_threshold == 0.005, f"Expected 0.005, got {det.collision_threshold}"
    print("  PASSED\n")


def test_maneuver_noise():
    """Maneuver engine should apply noise when noise_scale > 0."""
    print("=" * 60)
    print("TEST 5: Maneuver engine noise")
    print("=" * 60)
    from sim.maneuver_engine import ManeuverEngine, ManeuverType
    engine = ManeuverEngine()
    
    pos = np.array([6978.0, 0.0, 0.0])
    vel = np.array([0.0, 7.5, 0.0])
    
    # Without noise
    r1 = engine.apply_discrete_maneuver(
        pos, vel, ManeuverType.PROGRADE, fuel_available=1000.0, noise_scale=0.0
    )
    # With noise (run many times)
    np.random.seed(99)
    magnitudes = []
    for _ in range(50):
        r = engine.apply_discrete_maneuver(
            pos, vel, ManeuverType.PROGRADE, fuel_available=1000.0, noise_scale=0.10
        )
        magnitudes.append(r.delta_v_magnitude)
    
    std = np.std(magnitudes)
    print(f"  No-noise dv: {r1.delta_v_magnitude:.6f}")
    print(f"  Noisy dv std: {std:.6f} (should be > 0)")
    assert std > 0, "Noise should produce variation in ΔV magnitude"
    print("  PASSED\n")


def test_detailed_metrics():
    """compute_detailed_metrics should produce non-empty results."""
    print("=" * 60)
    print("TEST 6: compute_detailed_metrics")
    print("=" * 60)
    fake_stats = [
        {"total_collisions": 2, "total_fuel_used": 10.5, "total_maneuvers_executed": 5,
         "total_near_misses": 3, "min_separation_distance_km": 0.015, "tc8_active": True},
        {"total_collisions": 0, "total_fuel_used": 5.2, "total_maneuvers_executed": 2,
         "total_near_misses": 1, "min_separation_distance_km": 0.050, "tc8_active": False},
        {"total_collisions": 1, "total_fuel_used": 8.0, "total_maneuvers_executed": 3,
         "total_near_misses": 0, "min_separation_distance_km": 0.008, "tc8_active": True},
    ]
    metrics = compute_detailed_metrics(fake_stats)
    print(f"  Metrics keys: {sorted(metrics.keys())}")
    print(f"  collision_rate: {metrics['collision_rate']:.3f}")
    print(f"  near_miss_rate: {metrics['near_miss_rate']:.3f}")
    print(f"  maneuver_efficiency: {metrics['maneuver_efficiency']:.4f}")
    print(f"  mean_min_separation_m: {metrics['mean_min_separation_m']:.1f}")
    print(f"  tc8_difficulty_score: {metrics['tc8_difficulty_score']:.3f}")
    assert metrics["collision_rate"] > 0, "Should detect collisions"
    assert metrics["near_miss_rate"] > 0, "Should detect near misses"
    print("  PASSED\n")


if __name__ == "__main__":
    test_realism_disabled()
    test_realism_enabled()
    test_tc8_hardened()
    test_conjunction_detector_threshold()
    test_maneuver_noise()
    test_detailed_metrics()
    print("=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
