#!/usr/bin/env python3
"""
ADVANCED EXAMPLE: Safety Filter Demonstration and Scalability Testing
Shows:
- Control Barrier Function safety filter in action
- Impact of safety filter on collision avoidance
- Performance scaling with number of agents
- Detailed metrics and logging
"""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from sim.simulator import SimulationRunner
from sim.orbit_propagator import OrbitPropagator
from sim.conjunction_detector import ConjunctionDetector
import matplotlib.pyplot as plt
import pandas as pd


def demo_safety_filter_effectiveness():
    """
    Demonstrate effectiveness of CBF safety filter.
    Compare with and without safety filter.
    """
    print("\n" + "="*70)
    print("ADVANCED DEMO 1: Safety Filter Effectiveness")
    print("="*70)
    
    scenarios = [
        {'num_sats': 5, 'num_debris': 20, 'use_filter': False},
        {'num_sats': 5, 'num_debris': 20, 'use_filter': True},
    ]
    
    results = []
    
    for scenario in scenarios:
        print(f"\nTesting: {scenario['num_sats']} satellites, "
              f"{scenario['num_debris']} debris, "
              f"Safety filter: {scenario['use_filter']}")
        
        runner = SimulationRunner(
            num_satellites=scenario['num_sats'],
            num_debris=scenario['num_debris'],
            use_safety_filter=scenario['use_filter'],
            policy_type='baseline',
            enable_logging=True
        )
        
        stats_list = runner.run_multiple_episodes(
            num_episodes=3,
            max_steps=500,
            verbose=False
        )
        
        aggregated = runner._aggregate_stats(stats_list)
        aggregated['use_filter'] = scenario['use_filter']
        results.append(aggregated)
        
        print(f"  Mean collisions: {aggregated['mean_collisions']:.2f}")
        print(f"  Mean fuel: {aggregated['mean_fuel']:.2f} kg")
        print(f"  Success rate: {aggregated['success_rate']*100:.1f}%")
    
    # Compare
    print("\n" + "-"*70)
    print("SAFETY FILTER IMPACT:")
    without = results[0]
    with_filter = results[1]
    
    collision_reduction = (without['mean_collisions'] - with_filter['mean_collisions']) / \
                         (without['mean_collisions'] + 1e-6) * 100
    
    print(f"  Collision reduction: {collision_reduction:+.1f}%")
    print(f"  Fuel overhead: {(with_filter['mean_fuel'] - without['mean_fuel']):.2f} kg "
          f"({((with_filter['mean_fuel'] - without['mean_fuel']) / without['mean_fuel'] * 100):.1f}%)")
    print()


def demo_scalability():
    """
    Test scalability with increasing number of objects.
    """
    print("\n" + "="*70)
    print("ADVANCED DEMO 2: Scalability Analysis")
    print("="*70)
    
    configs = [
        {'num_sats': 3, 'num_debris': 10},
        {'num_sats': 5, 'num_debris': 50},
        {'num_sats': 10, 'num_debris': 100},
    ]
    
    scalability_results = []
    
    for config in configs:
        print(f"\nConfig: {config['num_sats']} satellites, {config['num_debris']} debris")
        
        runner = SimulationRunner(
            num_satellites=config['num_sats'],
            num_debris=config['num_debris'],
            use_safety_filter=True,
            policy_type='rule_based',
            enable_logging=False
        )
        
        stats_list = runner.run_multiple_episodes(
            num_episodes=2,
            max_steps=300,
            verbose=False
        )
        
        aggregated = runner._aggregate_stats(stats_list)
        
        result = {
            'num_sats': config['num_sats'],
            'num_debris': config['num_debris'],
            'total_objects': config['num_sats'] + config['num_debris'],
            **aggregated
        }
        scalability_results.append(result)
        
        print(f"  Mean collisions: {aggregated['mean_collisions']:.2f}")
        print(f"  Mean fuel: {aggregated['mean_fuel']:.2f} kg")
    
    # Print table
    print("\n" + "-"*70)
    print("SCALABILITY SUMMARY:")
    print("-"*70)
    df = pd.DataFrame(scalability_results)
    print(df[['num_sats', 'num_debris', 'total_objects', 'mean_collisions', 'mean_fuel', 'success_rate']].to_string(index=False))
    print()


def demo_orbit_propagation():
    """
    Demonstrate SGP4 orbit propagation accuracy.
    """
    print("\n" + "="*70)
    print("ADVANCED DEMO 3: Orbit Propagation (SGP4)")
    print("="*70)
    
    propagator = OrbitPropagator()
    
    # Create sample satellites
    print("\nCreating sample satellites in LEO...")
    for i in range(3):
        propagator.generate_sample_tle(
            f"SAT_{i}",
            semi_major_axis_km=6800 + i * 100,
            inclination_deg=51.6 + i * 0.2
        )
    
    # Propagate over time
    from datetime import datetime, timedelta
    
    start_time = datetime.utcnow()
    propagation_times = [start_time + timedelta(minutes=t) for t in range(0, 100, 10)]
    
    print(f"\nPropagating {len(propagator.satellites)} satellites for {len(propagation_times)} timesteps...")
    
    for obj_id in list(propagator.satellites.keys())[:1]:  # Just first satellite
        positions = []
        
        for t in propagation_times:
            state = propagator.propagate(obj_id, t)
            positions.append(state.position)
        
        positions = np.array(positions)
        
        # Compute orbital statistics
        distances_from_earth = np.linalg.norm(positions, axis=1)
        mean_distance = np.mean(distances_from_earth)
        altitude = mean_distance - 6371  # Earth radius
        
        print(f"\n{obj_id} Statistics:")
        print(f"  Mean altitude: {altitude:.2f} km")
        print(f"  Altitude variation: {np.std(distances_from_earth):.3f} km")
        print(f"  Trajectory length: {np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1)):.2f} km")


def demo_conjunction_detection():
    """
    Demonstrate conjunction detection and risk scoring.
    """
    print("\n" + "="*70)
    print("ADVANCED DEMO 4: Conjunction Detection & Risk Scoring")
    print("="*70)
    
    detector = ConjunctionDetector(
        distance_threshold_km=20.0,
        collision_threshold_km=0.025
    )
    
    propagator = OrbitPropagator()
    
    # Create objects with potential close approach
    print("\nCreating objects with designed close approach...")
    
    # Satellite 1: nominal orbit
    propagator.generate_sample_tle("SAT_1", semi_major_axis_km=6800)
    
    # Satellite 2: slightly different orbit (will have close approach)
    propagator.generate_sample_tle("SAT_2", semi_major_axis_km=6805, inclination_deg=51.65)
    
    # Propagate and detect
    from datetime import datetime, timedelta
    
    current_time = datetime.utcnow()
    detections = []
    
    print(f"\nDetecting conjunctions over 24 hours...")
    
    for hour in range(0, 24, 1):
        t = current_time + timedelta(hours=hour)
        
        states = propagator.propagate_all(t)
        
        # Convert to dict format
        object_states = {obj_id: state.to_array() for obj_id, state in states.items()}
        
        # Detect
        alerts = detector.detect(object_states, t)
        
        if alerts:
            for alert in alerts:
                detections.append({
                    'hour': hour,
                    'distance_km': alert.distance_km,
                    'risk_score': alert.risk_score,
                    'time_to_tca': alert.time_to_closest_approach_s / 3600  # convert to hours
                })
                
                if alert.risk_score > 0.5:
                    print(f"  Hour {hour}: HIGH RISK conjunction detected!")
                    print(f"    Distance: {alert.distance_km:.3f} km")
                    print(f"    Risk score: {alert.risk_score:.3f}")
                    print(f"    Time to TCA: {alert.time_to_closest_approach_s/60:.1f} min")
    
    if detections:
        df = pd.DataFrame(detections)
        print(f"\nDetected {len(detections)} conjunction events")
        print(f"Max risk score: {df['risk_score'].max():.3f}")
    else:
        print("\nNo conjunctions detected (objects too far apart)")


def main():
    """Advanced example is not part of minimal core collision avoidance code."""
    print("Advanced example removed. Use `python main.py --demo` or `demo.py` for core collision avoidance functionality.")


if __name__ == '__main__':
    main()
        ("Conjunction Detection", demo_conjunction_detection),
    ]
    
    for i, (name, demo_func) in enumerate(demos, 1):
        try:
            demo_func()
        except Exception as e:
            print(f"\n✗ Error in {name}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*70)
    print("All demonstrations completed!")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()


    def _handle_coordination_request(self, message: Dict[str, Any]) -> None:
        """Handle coordination requests from satellites."""
        payload = message.get("payload", {})
        satellite_id = message.get("sender_id")
        
        # Coordinate with other satellites
        print(f"  [COORD] {self.name} coordinating for {satellite_id}")
        
        # Inform all other satellites about the issue
        for peer_id in self.peers:
            if peer_id != satellite_id:
                self.send_message(
                    peer_id,
                    "COORDINATION_ALERT",
                    {
                        "affected_satellite": satellite_id,
                        "issue_description": payload.get("issue")
                    }
                )

    def _respond_to_heartbeat(self, message: Dict[str, Any]) -> None:
        """Respond to health monitor heartbeat."""
        self.send_message(
            message.get("sender_id"),
            "HEARTBEAT_RESPONSE",
            {
                "agent_id": self.agent_id,
                "is_alive": self.is_alive,
                "health_status": self.health_status.value,
                "coordination_level": self.coordination_level
            }
        )

    def execute_routine(self) -> None:
        """Execute controller routine."""
        self.process_messages()
        
        # Check if coordination is needed
        unhealthy_count = sum(
            1 for agent in self.peers.values()
            if agent.health_status in [HealthStatus.UNHEALTHY, HealthStatus.DEGRADED]
        )
        
        if unhealthy_count > 1:
            print(f"  [WARN] {self.name} detected {unhealthy_count} struggling agents, increasing coordination")
            self.coordination_level = min(2.0, self.coordination_level + 0.2)
        else:
            self.coordination_level = max(1.0, self.coordination_level - 0.1)


class ResilientSatelliteAgent(SatelliteAgent):
    """
    Enhanced satellite agent with additional resilience features.
    Can temporarily boost performance if needed.
    """

    def __init__(self, agent_id: str, name: str):
        super().__init__(agent_id, name)
        self.boost_active = False
        self.boost_duration = 0

    def execute_routine(self) -> None:
        """Execute enhanced routine with boost capability."""
        super().execute_routine()
        
        # Check if boost is active
        if self.boost_active:
            self.boost_duration -= 1
            if self.boost_duration <= 0:
                self.boost_active = False
                print(f"  ⚡ {self.name} boost deactivated")
            else:
                # Boost reduces degradation
                self.degrade_performance(1.0)

    def activate_emergency_boost(self, duration: int = 3) -> None:
        """
        Activate performance boost for emergency situations.
        
        Args:
            duration: Number of cycles to maintain boost
        """
        if not self.boost_active:
            self.boost_active = True
            self.boost_duration = duration
            print(f"  ⚡ {self.name} activated emergency boost!")


def run_advanced_simulation():
    """Run an advanced simulation with mixed agent types."""
    
    print("=" * 70)
    print("[ADVANCED AGENTIC AI SIMULATION]")
    print("=" * 70)
    
    # Create orchestrator
    orchestrator = AgentOrchestrator("Advanced Multi-Agent System")
    
    # Create standard satellites
    satellite1 = SatelliteAgent(str(uuid.uuid4())[:8], "Satellite-Standard-1")
    satellite2 = SatelliteAgent(str(uuid.uuid4())[:8], "Satellite-Standard-2")
    
    # Create resilient satellites
    satellite3 = ResilientSatelliteAgent(str(uuid.uuid4())[:8], "Satellite-Resilient-1")
    satellite4 = ResilientSatelliteAgent(str(uuid.uuid4())[:8], "Satellite-Resilient-2")
    
    # Create controller agent
    controller = ControllerAgent(str(uuid.uuid4())[:8], "Controller-1")
    
    # Create health monitor
    monitor = HealthMonitorAgent(str(uuid.uuid4())[:8], "Monitor-1")
    
    # Register all agents
    orchestrator.register_agent(satellite1)
    orchestrator.register_agent(satellite2)
    orchestrator.register_agent(satellite3)
    orchestrator.register_agent(satellite4)
    orchestrator.register_agent(controller)
    orchestrator.register_agent(monitor)
    
    # Setup network
    orchestrator.setup_agent_network()
    
    # Run simulation
    orchestrator.run_simulation(max_iterations=20)


if __name__ == "__main__":
    run_advanced_simulation()
