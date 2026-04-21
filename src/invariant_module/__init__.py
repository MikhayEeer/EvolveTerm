"""Invariant inference and instrumentation module.

This package owns invariant-specific implementation. The main
``evolve_term`` package should call into this package instead of keeping
invariant logic inline.
"""

from .predictor import InvariantPredictor
from .io import parse_invariants_content
from .agents import (
    BoundaryOpennessAgent,
    ControlFlowCoverageAgent,
    MissingConstantAgent,
    MultiAgentInvariantOptimizer,
)
from .houdini import HoudiniCheckResult, HoudiniFilter
from .refinement_pipeline import InvariantRefinementPipeline, InvariantRefinementResult

__all__ = [
    "BoundaryOpennessAgent",
    "ControlFlowCoverageAgent",
    "HoudiniCheckResult",
    "HoudiniFilter",
    "InvariantPredictor",
    "InvariantRefinementPipeline",
    "InvariantRefinementResult",
    "MissingConstantAgent",
    "MultiAgentInvariantOptimizer",
    "parse_invariants_content",
]
