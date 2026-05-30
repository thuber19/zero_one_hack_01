"""
physics — physical world model for semiconductor process sequences.

Modules
-------
ontology      : step → category mapping and physics feature vectors
state_machine : wafer state dataclass + step-by-step state transitions + validator
"""
from physics.ontology import classify_step, step_physics_vector, STEP_CATEGORY
from physics.state_machine import (
    WaferState,
    PhysicsViolation,
    apply_step,
    validate_by_state_machine,
)

__all__ = [
    "classify_step",
    "step_physics_vector",
    "STEP_CATEGORY",
    "WaferState",
    "PhysicsViolation",
    "apply_step",
    "validate_by_state_machine",
]
