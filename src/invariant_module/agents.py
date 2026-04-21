from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, List

from .houdini import dedupe_preserve_order
from .parsing import parse_acsl_invariants, parse_llm_yaml


@dataclass
class AgentResult:
    name: str
    invariants: List[str]
    raw_response: str = ""


class LLMInvariantAgent:
    name = "base"
    system_prompt = ""
    user_prompt = ""

    def __init__(self, llm_client: Any):
        self.llm_client = llm_client

    def refine(self, code: str, invariants: List[str]) -> AgentResult:
        prompt = {
            "system": self.system_prompt,
            "user": self.user_prompt.format(
                code=code,
                invariants=json.dumps(invariants, ensure_ascii=False, indent=2),
            ),
            "max_tokens": 4096,
        }
        response = self.llm_client.complete(prompt)
        parsed = self._parse_invariants(response)
        if not parsed:
            parsed = invariants
        return AgentResult(self.name, dedupe_preserve_order(parsed), response)

    def _parse_invariants(self, response: str) -> List[str]:
        data = parse_llm_yaml(response)
        if isinstance(data, dict):
            invs = data.get("invariants") or data.get("loop_invariants") or []
            if isinstance(invs, list):
                return [str(item) for item in invs if str(item).strip()]
        if isinstance(data, list):
            return [str(item) for item in data if str(item).strip()]
        return parse_acsl_invariants(response)


class MissingConstantAgent(LLMInvariantAgent):
    name = "missing_constant_agent"
    system_prompt = (
        "You are a C loop invariant refinement agent. Your task is to detect "
        "missing constants and bounds in the current invariant set, such as "
        "initial constants, array bounds, counter limits, and variable ranges. "
        "Return only YAML with key 'invariants'."
    )
    user_prompt = """C code:
{code}

Current invariants:
{invariants}

Refine the invariant set by adding any missing constant or bound constraints.
Keep valid existing invariants. Return:
invariants:
  - <C boolean expression>
"""


class BoundaryOpennessAgent(LLMInvariantAgent):
    name = "boundary_openness_agent"
    system_prompt = (
        "You are a C loop invariant refinement agent. Your task is to check "
        "whether each numeric boundary is open or closed, for example > vs >= "
        "and < vs <=. Return only YAML with key 'invariants'."
    )
    user_prompt = """C code:
{code}

Current invariants:
{invariants}

Refine comparison operators so boundary openness is correct for the loop.
Keep valid existing invariants. Return:
invariants:
  - <C boolean expression>
"""


class ControlFlowCoverageAgent(LLMInvariantAgent):
    name = "control_flow_coverage_agent"
    system_prompt = (
        "You are a C loop invariant refinement agent. Your task is to check "
        "whether the invariant set holds across all control-flow paths, "
        "including if/else branches, continue, break-adjacent updates, and "
        "nested updates. Return only YAML with key 'invariants'."
    )
    user_prompt = """C code:
{code}

Current invariants:
{invariants}

Refine the invariant set so it accounts for all relevant control-flow paths.
Add missing path conditions or branch-stable facts when needed. Return:
invariants:
  - <C boolean expression>
"""


class MultiAgentInvariantOptimizer:
    def __init__(self, agents: List[LLMInvariantAgent]):
        self.agents = agents
        self.last_results: List[AgentResult] = []

    def optimize(self, code: str, invariants: List[str]) -> List[str]:
        current = dedupe_preserve_order(invariants)
        self.last_results = []
        for agent in self.agents:
            result = agent.refine(code, current)
            self.last_results.append(result)
            current = dedupe_preserve_order(result.invariants)
        return current


def build_default_optimizer(llm_client: Any) -> MultiAgentInvariantOptimizer:
    return MultiAgentInvariantOptimizer(
        [
            MissingConstantAgent(llm_client),
            BoundaryOpennessAgent(llm_client),
            ControlFlowCoverageAgent(llm_client),
        ]
    )
