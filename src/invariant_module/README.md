# Invariant Module Flow

This module is organized around the invariant research pipeline below:

1. Two-stage LLM candidate generation
   - `inv_assume/strategies/two_stage.py`
   - `TwoStageStrategy.generate_candidates(code)` returns a candidate invariant set.
   - `TwoStageStrategy.generate(code)` is kept for backward compatibility and returns the first candidate.

2. Houdini filtering
   - `houdini.py`
   - `HoudiniFilter` implements the iterative remove-refuted-candidates loop.
   - The verifier/checker is injected, so the same orchestration can use SeaHorn, Z3, Boogie, or a future custom checker.

3. Multi-agent refinement network
   - `agents.py`
   - `MissingConstantAgent`: checks missing constants and bounds.
   - `BoundaryOpennessAgent`: checks open/closed boundary operators, such as `>` vs `>=`.
   - `ControlFlowCoverageAgent`: checks whether invariants cover all relevant control-flow paths.

4. Final orchestration
   - `refinement_pipeline.py`
   - `InvariantRefinementPipeline` runs:
     `two-stage candidates -> Houdini -> missing constants agent -> boundary agent -> control-flow agent -> optional final Houdini`.

The current Houdini implementation is verifier-agnostic. To make it fully semantic,
provide a `HoudiniChecker` that checks the current invariant set and returns which
candidate invariants were refuted.
