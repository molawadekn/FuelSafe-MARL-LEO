# 📚 Technical Guide - FuelSafe

## Table of Contents
1. [Architecture Overview](#architecture-overview)
2. [Component Details](#component-details)
3. [Communication Protocol](#communication-protocol)
4. [Extending the System](#extending-the-system)
5. [Advanced Patterns](#advanced-patterns)
6. [Performance Considerations](#performance-considerations)

## Architecture Overview

### System Design

The Agentic AI system follows a **distributed agent architecture** where:

- **Agents are autonomous**: Each agent has its own state, lifecycle, and decision-making logic
- **Communication is asynchronous**: Agents communicate via message passing (not direct procedure calls)
- **System is decentralized**: No single point of control; decisions are made collaboratively
- **State is local**: Each agent maintains its own state; no shared global state

```
┌─────────────────────────────────────────────────────────┐
│           ORCHESTRATOR (Management Layer)                │
│  ├─ Agent Registration                                  │
│  ├─ Network Setup                                       │
│  ├─ Simulation Control                                  │
│  └─ Results Aggregation                                │
└─────────────────────────────────────────────────────────┘
                          ▲
                    ▼     │     ▼
        ┌───────────┬───────────┬───────────┐
        │ SATELLITE │ SATELLITE │ SATELLITE │
        │  AGENT 1  │  AGENT 2  │  AGENT 3  │
        └───────────┴───────────┴───────────┘
                          ▲
                          │
        ┌─────────────────┴─────────────────┐
        │   HEALTH MONITOR AGENT            │
        │   (System-wide Observer)          │
        └───────────────────────────────────┘

MESSAGE FLOW:
  Satellite A ──→ Message Queue ──→ Satellite B
       │                                ▲
       └──→ Health Monitor ─────────────┘
```

## Component Details

### 1. Agent Base Class

**File**: `src/agent.py`

**Key Methods**:
```python
def send_message(recipient_id, message_type, payload)
    # Sends a message to another agent

def receive_message(message)
    # Receives a message from another agent

def broadcast_message(message_type, payload)
    # Sends message to all peers

def process_messages()
    # Processes all queued messages

def handle_message(message) [ABSTRACT]
    # Override to handle specific message types

def execute_routine() [ABSTRACT]
    # Override to implement agent behavior
```

**State**:
- `agent_id`: Unique identifier
- `health_status`: Current health (HEALTHY, DEGRADED, UNHEALTHY, DEAD)
- `performance_metric`: 0-100% health indicator
- `message_queue`: Pending messages
- `peers`: Connected agents dictionary

### 2. Satellite Agent

**File**: `src/satellite_agent.py`

**Capabilities**:
- Self-monitoring: Detects own health degradation
- Peer repair requests: Can request help from healthier peers
- Repair execution: Can repair other satellites
- Decision-making: Decides whether to repair requests based on comparative health

**Key Methods**:
```python
def execute_routine()
    # Main satellite logic:
    # 1. Process messages
    # 2. Simulate performance degradation
    # 3. Request repairs if unhealthy
    # 4. Update health status

def repair_itself(repair_amount)
    # Improve performance by repair_amount

def _should_repair_peer(peer_id)
    # Decide if should repair a peer based on:
    # - This agent's health status
    # - Comparative health with peer
    # - Random decision factor
```

**Message Handlers**:
- `REPAIR_REQUEST`: Handle incoming repair requests
- `REPAIR_DECISION`: Handle repair decision responses
- `HEARTBEAT_CHECK`: Respond to monitor checks
- `STATUS_QUERY`: Provide status information

### 3. Health Monitor Agent

**File**: `src/monitor_agent.py`

**Responsibilities**:
- Heartbeat checking: Verify all agents are alive
- Status collection: Gather detailed health information
- Health history: Maintain comprehensive logs
- Alerting: Detect and report anomalies

**Key Methods**:
```python
def execute_routine()
    # Monitor logic:
    # 1. Process messages
    # 2. Send heartbeat checks
    # 3. Query detailed status
    # 4. Log health summary

def get_health_report()
    # Return comprehensive health analysis

def _log_health_summary()
    # Create and store health snapshot
```

**Health Records**:
```python
{
    "agent_id": "unique_id",
    "is_alive": bool,
    "health_status": str,  # healthy, degraded, unhealthy, dead
    "last_heartbeat": ISO8601_timestamp,
    "heartbeat_count": int,
    "updated_at": ISO8601_timestamp
}
```

### 4. Orchestrator

**File**: `src/orchestrator.py`

**Responsibilities**:
- Agent lifecycle management
- Network topology setup
- Simulation execution
- Results aggregation

**Key Methods**:
```python
def register_agent(agent)
    # Add agent to system

def setup_agent_network()
    # Establish peer connections

def execute_step()
    # Run one simulation cycle

def run_simulation(max_iterations)
    # Run complete simulation

def get_system_status()
    # Get snapshot of all agents
```

## Communication Protocol

### Message Format

All messages follow this structure:
```json
{
    "sender_id": "agent_id",
    "sender_name": "Satellite-Alpha",
    "recipient_id": "target_agent_id",
    "message_type": "REPAIR_REQUEST",
    "payload": {
        "key": "value",
        ...
    },
    "timestamp": "2026-03-05T19:01:08.832134",
    "message_id": "unique_uuid"
}
```

### Message Types

#### HEARTBEAT_CHECK / HEARTBEAT_RESPONSE
**Direction**: Monitor → Satellites / Satellites → Monitor
**Purpose**: Verify agent is alive and respond with health status
```python
# Request
{"check_time": ISO8601_timestamp, "sequence_number": int}

# Response
{
    "agent_id": str,
    "is_alive": bool,
    "health_status": str
}
```

#### REPAIR_REQUEST / REPAIR_DECISION
**Direction**: Unhealthy Satellite → Healthy Peer / Peer → Requester
**Purpose**: Request mutual repair and respond with decision
```python
# Request
{
    "requester_id": str,
    "requester_health": str,
    "performance_metric": float
}

# Decision
{
    "accepted": bool,
    "repairer_id": str,
    "repairer_name": str,
    "repair_amount": float,  # if accepted
    "reason": str             # if rejected
}
```

#### STATUS_QUERY / STATUS_RESPONSE
**Direction**: Monitor → Satellites / Satellites → Monitor
**Purpose**: Request detailed agent status
```python
# Response
{
    "agent_id": str,
    "name": str,
    "type": str,
    "health_status": str,
    "is_alive": bool,
    "performance_metric": float,
    "repair_count": int,
    "last_heartbeat": ISO8601_timestamp,
    "pending_messages": int
}
```

## Extending the System

### Creating Custom Agent Types

```python
from src.agent import Agent, AgentType

class CustomAgent(Agent):
    def __init__(self, agent_id: str, name: str):
        super().__init__(agent_id, AgentType.CUSTOM, name)
        # Initialize custom state
        self.custom_state = {}
    
    def handle_message(self, message: Dict[str, Any]) -> None:
        """Handle messages specific to this agent type."""
        message_type = message.get("message_type")
        
        if message_type == "CUSTOM_MESSAGE":
            self._handle_custom_message(message)
    
    def execute_routine(self) -> None:
        """Implement custom behavior."""
        # Process messages
        self.process_messages()
        
        # Custom logic
        self._perform_custom_action()
    
    def _perform_custom_action(self) -> None:
        # Implement your custom logic
        pass
```

### Adding New Message Types

1. **Define message handling in agent**:
```python
def handle_message(self, message):
    if message.get("message_type") == "MY_NEW_TYPE":
        self._handle_my_new_type(message)

def _handle_my_new_type(self, message):
    # Implementation
    pass
```

2. **Send new message type**:
```python
self.send_message(
    recipient_id,
    "MY_NEW_TYPE",
    {"key": "value"}
)
```

3. **Broadcast to all peers**:
```python
self.broadcast_message(
    "MY_NEW_TYPE",
    {"key": "value"}
)
```

### Custom Agent Behavior

```python
class SmartSatelliteAgent(SatelliteAgent):
    """Enhanced satellite with predictive repair."""
    
    def __init__(self, agent_id: str, name: str):
        super().__init__(agent_id, name)
        self.health_history = []
    
    def execute_routine(self) -> None:
        # Call parent routine
        super().execute_routine()
        
        # Add predictive logic
        self._predict_failure()
        self._proactive_repair()
    
    def _predict_failure(self) -> None:
        """Predict when agent will become unhealthy."""
        self.health_history.append(self.performance_metric)
        
        if len(self.health_history) >= 3:
            # Calculate degradation trend
            recent = self.health_history[-3:]
            trend = sum(recent) / len(recent)
            
            if trend < 50:
                print(f"⚠️  {self.name} predicts failure in next few cycles")
    
    def _proactive_repair(self) -> None:
        """Request repair before becoming unhealthy."""
        if self.health_status == HealthStatus.DEGRADED and not self.repair_decision_pending:
            self._request_peer_repair()
```

## Advanced Patterns

### 1. Hierarchical Control

Create a supervisor agent that manages multiple satellites:
```python
class SupervisorAgent(Agent):
    def __init__(self, agent_id, name):
        super().__init__(agent_id, AgentType.SUPERVISOR, name)
        self.managed_agents = []
    
    def execute_routine(self):
        self.process_messages()
        
        # Check status of managed agents
        for agent_id in self.managed_agents:
            agent = self.peers.get(agent_id)
            if agent and agent.health_status == HealthStatus.UNHEALTHY:
                self._escalate_repair(agent_id)
```

### 2. Cooperative Decision Making

Multiple agents voting on decisions:
```python
def execute_collective_repair(self, agent_requesting_repair):
    # Send vote request to all healthy peers
    self.broadcast_message(
        "REPAIR_VOTE",
        {
            "agent_needing_repair": agent_requesting_repair,
            "vote_timeout": 3
        }
    )
    
    # Collect votes from peers
    votes = []
    while len(self.message_queue) > 0:
        msg = self.message_queue.pop(0)
        if msg.get("message_type") == "REPAIR_VOTE_RESPONSE":
            votes.append(msg.get("payload", {}).get("vote"))
    
    # Execute if majority votes yes
    if votes.count(True) > len(votes) / 2:
        print(f"✅ Collective repair approved for {agent_requesting_repair}")
```

### 3. State Machine Pattern

Agents with distinct states:
```python
from enum import Enum

class AgentState(Enum):
    IDLE = "idle"
    REPAIRING = "repairing"
    DEGRADED = "degraded"
    EMERGENCY = "emergency"

class StatefulSatellite(SatelliteAgent):
    def __init__(self, agent_id, name):
        super().__init__(agent_id, name)
        self.current_state = AgentState.IDLE
    
    def execute_routine(self):
        if self.current_state == AgentState.IDLE:
            self._handle_idle_state()
        elif self.current_state == AgentState.REPAIRING:
            self._handle_repairing_state()
        elif self.current_state == AgentState.DEGRADED:
            self._handle_degraded_state()
        elif self.current_state == AgentState.EMERGENCY:
            self._handle_emergency_state()
```

## Performance Considerations

### Scalability

**Current System**:
- Tested with 3-8 agents
- Message complexity: O(n²) for broadcasts
- Memory: ~1KB per agent + message history

**Optimization for Larger Systems**:

1. **Message routing**: Instead of peer-to-peer, use message bus
```python
class MessageBus:
    def __init__(self):
        self.subscribers = {}
    
    def subscribe(self, agent_id, message_types):
        self.subscribers[agent_id] = message_types
    
    def publish(self, message_type, message):
        # Send to subscribed agents only
        pass
```

2. **Agent grouping**: Organize agents in clusters
```python
class AgentCluster:
    def __init__(self):
        self.local_agents = {}
        self.clusters = {}
    
    def intra_cluster_communication(self):
        # Fast local communication
        pass
    
    def inter_cluster_communication(self):
        # Slower cross-cluster communication
        pass
```

3. **Lazy message processing**:
```python
def process_messages(self):
    # Process only most recent messages
    # Discard old ones
    if len(self.message_queue) > 100:
        self.message_queue = self.message_queue[-100:]
```

### Memory Management

- Limit message queue size
- Periodic cleanup of old health records
- Compress history using aggregation

### Simulation Speed

- Reduce `max_iterations`
- Increase `heartbeat_interval`
- Disable logging for production runs

---

**Version**: 1.0.0
**Last Updated**: March 5, 2026
