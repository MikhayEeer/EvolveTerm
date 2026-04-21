from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List
import yaml

try:
    from evolve_term.llm_client import APILLMClient
except ImportError:  # pragma: no cover - supports python -m src....
    from src.evolve_term.llm_client import APILLMClient

from ..utils import (
    syntax_filter,
    z3_filter,
    extract_modified_vars,
)

@dataclass(frozen=True)
class PromptPair:
    name: str
    system: str
    user: str

    def render(self, **replacements: str) -> Dict[str, str]:
        system = self.system
        user = self.user
        for key, value in replacements.items():
            placeholder = "{" + key + "}"
            system = system.replace(placeholder, value)
            user = user.replace(placeholder, value)
        return {"system": system, "user": user}

def load_prompt_pair(prompt_dir: Path, base_name: str) -> PromptPair:
    system_path = prompt_dir / f"{base_name}.system.txt"
    user_path = prompt_dir / f"{base_name}.user.txt"
    if not system_path.exists() or not user_path.exists():
        raise FileNotFoundError(f"Prompt files not found for '{base_name}' in {prompt_dir}")
    return PromptPair(
        name=base_name,
        system=system_path.read_text(encoding="utf-8"),
        user=user_path.read_text(encoding="utf-8"),
    )

class TwoStageStrategy:
    def __init__(self, llm_config: str = "llm_config.json"):
        self.llm_client = APILLMClient(config_name=llm_config)
        self.repo_root = self._find_repo_root()
        self.prompt_dir = self.repo_root / "prompts" / "invariants" / "2stage_simple"

    def _find_repo_root(self) -> Path:
        for parent in Path(__file__).resolve().parents:
            if (parent / "prompts" / "invariants").exists():
                return parent
        return Path(__file__).resolve().parents[4]

    def generate_candidates(self, loop_context: str) -> List[str]:
        """
        Executes the 2-stage generation process:
        1. Generate Atoms -> Filter
        2. Generate Candidate Invariants (using atoms) -> Filter
        3. Return the filtered candidate invariant set.
        """
        if not self.prompt_dir.exists():
             raise FileNotFoundError(f"Prompt directory not found: {self.prompt_dir}")

        # 1. Load Prompts
        stage1_prompts = load_prompt_pair(self.prompt_dir, "seahorn_stage1")
        stage2_prompts = load_prompt_pair(self.prompt_dir, "seahorn_stage2")

        # 2. Extract modified vars (simple regex based)
        modified_vars = extract_modified_vars(loop_context)

        # 3. Stage 1: Atoms
        s1_prompt = stage1_prompts.render(CODE=loop_context)
        s1_resp = self.llm_client.complete(s1_prompt)
        
        atoms = self._parse_list(s1_resp, "atoms")
        
        # Filter Atoms
        atoms, _ = syntax_filter(atoms, modified_vars)
        atoms, _ = z3_filter(atoms, max_keep=10)
        
        if not atoms:
            return ["true"]

        # 4. Stage 2: Candidates
        atoms_block = "\n".join(f"- {a}" for a in atoms)
        s2_prompt = stage2_prompts.render(CODE=loop_context, ATOMS=atoms_block)
        s2_resp = self.llm_client.complete(s2_prompt)
        
        invariants = self._parse_list(s2_resp, "invariants")
        
        # Filter Candidates
        invariants, _ = syntax_filter(invariants, modified_vars)
        invariants, _ = z3_filter(invariants, max_keep=5)
        
        if not invariants:
             return ["true"]

        return invariants

    def generate(self, loop_context: str) -> str:
        """
        Backward-compatible one-shot API used by the old inv_assume pipeline.
        """
        candidates = self.generate_candidates(loop_context)
        return candidates[0] if candidates else "true"

    def _parse_list(self, text: str, key: str) -> List[str]:
        # Helper to parse YAML list from response
        try:
            # Try to find YAML block
            if "```yaml" in text:
                text = text.split("```yaml")[1].split("```")[0]
            elif "```" in text:
                text = text.split("```")[1].split("```")[0]
            
            data = yaml.safe_load(text)
            if isinstance(data, dict):
                 items = data.get(key, [])
                 if isinstance(items, list): return items
            if isinstance(data, list): return data
            return []
        except:
            return []
