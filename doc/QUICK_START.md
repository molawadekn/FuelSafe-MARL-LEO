# 🚀 Quick Start Guide - FuelSafe

## 5-Minute Setup

### 1. Prerequisites
- Python 3.8+
- Windows/Mac/Linux

### 2. Installation

```bash
# Navigate to project
cd "c:\Users\molaw\code\Final Year Project\FuelSafe-MARL-LEO"

# Optional: Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### 3. Run Demo

```bash
python main.py --demo
```

You should see:
- 2 policy scenarios running (baseline and rule-based)
- 2 episodes each with collision/fuel metrics
- Final comparison showing which policy performs better

**Output Example:**
```
================================================================================
Fuel-Constrained Multi-Agent Collision Avoidance Demonstration
================================================================================

Scenario: 3 Satellites + 5 Debris Objects
Duration: 2.7 hours (~1 orbit)
Episodes: 3

================================================================================
Testing: BASELINE Policy
================================================================================

Episode 1/3:
  Step 100/1000 - Collisions: 0, Fuel: ..., Alerts: ...
  Final Stats:
    - Total Collisions: 1
    - Total Fuel Used: 42.50 kg
    - Total Alerts: 24
    - Episode Length: 850 steps

[Results for all episodes...]

✓ WINNER: RULE_BASED policy with 0.33 mean collisions
```

## 📚 Example Commands

### Run Core Demo
```bash
python main.py --demo
```

No other command-line modes are included in the minimal core version.

## 🔍 Understanding Output

### Collision Count
- **0**: Perfect simulation, no collisions
- **>0**: Number of times objects crashed
- Goal: Minimize

### Fuel Used (kg)
- ΔV maneuvers consume fuel
- More aggressive maneuvers = more fuel
- Goal: Minimize while avoiding collisions

### Success Rate (%)
- Percentage of episodes with 0 collisions
- 100% = always safe
- Goal: Maximize

### Alerts
- Number of conjunction warnings triggered
- High alerts + low collisions = good detection and response

## 🎯 What Each Module Does

| Run | What You Get |
|-----|--------------|
| `--demo` | Quick comparison (2 policies, ~2 min) |
| `--compare` | Detailed policy analysis (3 policies, ~10 min) |
| `--experiment` | Scalability testing (3×2 configs, ~5 min) |
| `--train-marl` | Learn optimal policy (5 episodes, ~3 min) |

## 💡 Next Steps

### Try Custom Scenarios
```python
from sim.simulator import SimulationRunner

# Your custom scenario
runner = SimulationRunner(
    num_satellites=10,      # More satellites
    num_debris=100,         # More debris
    use_safety_filter=True, # Enable safety
    policy_type='rule_based'
)

stats = runner.run_episode(max_steps=2000)
```

## Customization

### Change Number of Agents

**In `main.py`**:
```python
# Create 5 satellites instead of 3
for i in range(5):
    satellite = create_satellite_agent(f"Satellite-{i}")
    orchestrator.register_agent(satellite)
```

### Change Health Thresholds

**In `src/agent.py`**:
```python
def update_health_status(self):
    if self.performance_metric >= 90:  # Changed from 85
        self.health_status = HealthStatus.HEALTHY
```

### Change Degradation Rate

**In `src/satellite_agent.py`**:
```python
# More frequent degradation
if random.random() > 0.3:  # Was 0.6 (now 70% instead of 40%)
    degradation = random.uniform(2.0, 4.0)  # Larger amounts
    self.degrade_performance(degradation)
```

### Change Simulation Length

**In `main.py`**:
```python
# Run for 30 steps instead of 15
orchestrator.run_simulation(max_iterations=30)
```

## Using Configuration

### Load Preset Configuration

```python
from config import PresetConfigurations

# Use stable system preset
config = PresetConfigurations.stable_system()

# Modify and use
config.simulation.num_satellite_agents = 5
config.simulation.max_iterations = 20
```

### View Configuration

```bash
python config.py
```

## Project Structure

```
Agentic_AI_POC/
├── main.py                 # Basic simulation (START HERE)
├── advanced_example.py     # Advanced features demo
├── test_scenarios.py       # Multiple test scenarios
├── config.py              # Configuration presets
├── utils.py               # Utility functions
├── src/
│   ├── __init__.py
│   ├── agent.py           # Base Agent class
│   ├── satellite_agent.py # Satellite implementation
│   ├── monitor_agent.py   # Monitor implementation
│   └── orchestrator.py    # System orchestrator
├── requirements.txt       # Dependencies
├── README.md             # Full documentation
├── TECHNICAL_GUIDE.md    # Architecture details
└── QUICK_START.md        # This file
```

## Key Files to Understand

1. **main.py** (50 lines)
   - Entry point for basic simulation
   - Shows how to create and run agents

2. **src/agent.py** (200 lines)
   - Core agent functionality
   - Base class for all agents

3. **src/satellite_agent.py** (150 lines)
   - Autonomous satellite agents
   - Repair decision logic

4. **src/monitor_agent.py** (150 lines)
   - Health monitoring logic
   - Heartbeat mechanism

5. **src/orchestrator.py** (150 lines)
   - System management
   - Simulation execution

## Common Tasks

### Monitor a Specific Agent

```python
from main import main
import uuid

# Create agent
sat = create_satellite_agent("Monitored-Sat")
print(f"Agent ID: {sat.agent_id}")
print(f"Initial Health: {sat.performance_metric}%")

# Run simulation and check final state
sat.execute_routine()
print(f"Final Health: {sat.performance_metric}%")
print(f"Final Status: {sat.health_status.value}")
```

### Log Simulation to File

```python
from utils import SimulationLogger

logger = SimulationLogger("my_simulation.json")

# During simulation
logger.log_agent_status("agent1", {"health": "healthy"})
logger.log_repair("agent1", "agent2", True)

# After simulation
logger.save_to_file()
logger.print_summary()
```

### Analyze Results

```python
from utils import SystemAnalyzer

# After simulation with monitor agent
analysis = SystemAnalyzer.analyze_agent_health(
    monitor.status_log
)

# View per-agent analysis
for agent_id, metrics in analysis.items():
    print(f"Agent {agent_id}: {metrics}")

# Calculate system health score
score = SystemAnalyzer.calculate_system_health_score(
    monitor.status_log
)
print(f"Overall System Health: {score:.1f}%")
```

## Troubleshooting

### No Output
- Check Python version: `python --version` (need 3.8+)
- Verify file permissions
- Try: `python -u main.py` (unbuffered output)

### Agents All Dead Immediately
- Reduce initial degradation chance in satellite_agent.py
- Increase initial health: `satellite.performance_metric = 85.0`

### Memory Usage High
- Reduce `max_iterations` (less history)
- Disable logging: Set `print_health_summary = False` in config
- Limit message queue size

### Slow Simulation
- Fewer agents: Change `num_satellite_agents` in main.py
- Fewer iterations: `run_simulation(max_iterations=5)`
- Skip details: `verbose_output = False` in config

## Next Steps

1. **Understand Agents**: Read `src/agent.py` comments
2. **See Repair in Action**: Run `test_scenarios.py` with repair scenario
3. **Extend System**: Create custom agent in `advanced_example.py`
4. **Analyze Results**: Use `utils.py` functions
5. **Read Technical Guide**: `TECHNICAL_GUIDE.md` for deep dive

## Resources

- **Full Documentation**: README.md
- **Technical Details**: TECHNICAL_GUIDE.md
- **Code Examples**: advanced_example.py, test_scenarios.py
- **Configuration**: config.py
- **Utilities**: utils.py

## Getting Help

### Common Questions

**Q: How do agents decide who repairs?**
A: In `_should_repair_peer()` - decision based on comparative health + randomness.

**Q: Can agents refuse repairs?**
A: Yes, if they're unhealthy or not healthier than requester.

**Q: How to add new agent types?**
A: Inherit from `Agent` and implement `handle_message()` and `execute_routine()`.

**Q: How to customize messages?**
A: Add new message type handling in `handle_message()` method.

### Debugging

Add print statements:
```python
def execute_routine(self):
    print(f"DEBUG: {self.name} executing routine, health={self.performance_metric}")
    super().execute_routine()
```

Or use logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
logger.debug(f"Agent status: {self.health_status}")
```

---

**Ready to explore?** Start with:
```bash
python main.py
```

For detailed info, see README.md
For architecture details, see TECHNICAL_GUIDE.md
