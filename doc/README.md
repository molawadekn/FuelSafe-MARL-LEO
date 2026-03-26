# FuelSafe-MARL-LEO

Fuel-Constrained Multi-Agent Reinforcement Learning for Orbital Collision Avoidance in LEO.

A modular Python simulator for satellite collision avoidance with fuel constraints, featuring:
- SGP4 orbit propagation
- ESA CDM ingestion and conjunction detection
- Multi-agent RL environment
- Control Barrier Function safety filtering
- Policy comparison (baseline vs rule-based)

## Quick Start

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run demo:
   ```bash
   python main.py
   ```

## Modules

- `sim/` : orbit propagation, CDM loading, conjunction detection, simulator.
- `env/` : multi-agent orbital environment.
- `policies/` : baseline and rule-based policy interface.
- `safety/` : control barrier function safety filter.
- `marl/` : MAPPO trainer framework.
- `experiments/` : experiment runner for policy evaluation.

## Usage

- `python main.py` – Run core collision avoidance demo
- `python demo.py` – Run focused demo with detailed output
