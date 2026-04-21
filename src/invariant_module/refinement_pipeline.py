from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Protocol

from .agents import AgentResult, MultiAgentInvariantOptimizer, build_default_optimizer
from .houdini import HoudiniCheckResult, HoudiniFilter, dedupe_preserve_order


class CandidateGenerator(Protocol):
    def generate_candidates(self, loop_context: str) -> List[str]:
        ...


@dataclass
class InvariantRefinementTrace:
    candidates: List[str] = field(default_factory=list)
    after_houdini: List[str] = field(default_factory=list)
    agent_results: List[AgentResult] = field(default_factory=list)
    final_houdini_trace: List[HoudiniCheckResult] = field(default_factory=list)


@dataclass
class InvariantRefinementResult:
    invariants: List[str]
    trace: InvariantRefinementTrace


class InvariantRefinementPipeline:
    """
    Research pipeline for invariant inference:
    1. two-stage LLM candidate generation
    2. Houdini filtering
    3. multi-agent LLM refinement
    4. optional final Houdini filtering
    """

    def __init__(
        self,
        candidate_generator: CandidateGenerator,
        houdini_filter: HoudiniFilter | None = None,
        optimizer: MultiAgentInvariantOptimizer | None = None,
        llm_client: Any | None = None,
        final_houdini: bool = True,
    ):
        self.candidate_generator = candidate_generator
        self.houdini_filter = houdini_filter or HoudiniFilter()
        self.optimizer = optimizer or (
            build_default_optimizer(llm_client) if llm_client is not None else None
        )
        self.final_houdini = final_houdini

    def infer(self, code: str) -> InvariantRefinementResult:
        candidates = dedupe_preserve_order(self.candidate_generator.generate_candidates(code))
        after_houdini = self.houdini_filter.filter(code, candidates)

        if self.optimizer is None:
            trace = InvariantRefinementTrace(
                candidates=candidates,
                after_houdini=after_houdini,
                final_houdini_trace=list(self.houdini_filter.last_trace),
            )
            return InvariantRefinementResult(after_houdini, trace)

        optimized = self.optimizer.optimize(code, after_houdini)
        final = optimized
        final_trace = []
        if self.final_houdini:
            final = self.houdini_filter.filter(code, optimized)
            final_trace = list(self.houdini_filter.last_trace)

        trace = InvariantRefinementTrace(
            candidates=candidates,
            after_houdini=after_houdini,
            agent_results=list(self.optimizer.last_results),
            final_houdini_trace=final_trace,
        )
        return InvariantRefinementResult(final, trace)
