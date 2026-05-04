"""
MODULE: Curriculum Manager
Progressive stage manager for evaluating collision avoidance policies.
"""

from __future__ import annotations

from typing import Any, Dict


class CurriculumManager:
    def __init__(self):
        # Define curriculum stages from easier to harder.
        self.stages = [
            {
                "stage": 1,
                "name": "Stage 1: Long TCA, Low Density",
                "max_debris": 3,
                "tca_range": (1800.0, 3600.0),  # 30 min to 60 min
                "collision_threshold_km": 50.0,
                "success_threshold": 0.95,
            },
            {
                "stage": 2,
                "name": "Stage 2: Moderate Risk, Medium Density",
                "max_debris": 10,
                "tca_range": (600.0, 1800.0),  # 10 min to 30 min
                "collision_threshold_km": 10.0,
                "success_threshold": 0.90,
            },
            {
                "stage": 3,
                "name": "Stage 3: Short TCA, High Density (TC8)",
                "max_debris": 25,
                "tca_range": (0.0, 600.0),  # up to 10 min
                "collision_threshold_km": 1.0,
                "success_threshold": 0.85,
            },
        ]
        self.current_stage_idx = 0
        self.history: list[float] = []

    def get_current_config(self) -> Dict[str, Any]:
        """Get environment configuration for the current operational stage."""
        return self.stages[self.current_stage_idx]

    def update_performance(self, success_rate: float) -> bool:
        """
        Record performance and attempt to advance stage if threshold met.
        Returns True if advanced to a new stage.
        """
        self.history.append(float(success_rate))

        # Require 3 consecutive performances above threshold to advance.
        if len(self.history) >= 3:
            recent_perf = self.history[-3:]
            threshold = float(self.stages[self.current_stage_idx]["success_threshold"])

            if all(p >= threshold for p in recent_perf):
                if self.current_stage_idx < len(self.stages) - 1:
                    self.current_stage_idx += 1
                    self.history = []
                    print(
                        f"\nCURRICULUM ADVANCED: "
                        f"{self.stages[self.current_stage_idx]['name']}\n"
                    )
                    return True
        return False

    def is_finished(self) -> bool:
        """Check if all stages are mastered."""
        return (
            self.current_stage_idx == len(self.stages) - 1
            and len(self.history) >= 3
            and all(p >= self.stages[-1]["success_threshold"] for p in self.history[-3:])
        )
