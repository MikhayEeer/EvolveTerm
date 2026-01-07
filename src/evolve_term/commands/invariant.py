"""Handler for the 'invariant' command."""
from __future__ import annotations
import json
import yaml
import sys
import re
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime
from rich.console import Console
import typer

from ..predict import Predictor
from ..prompts_loader import PromptRepository
from ..llm_client import build_llm_client
from ..cli_utils import collect_files, load_references, validate_yaml_required_keys
from ..utils import LiteralDumper

console = Console()

class InvariantHandler:
    def __init__(self, llm_config: str):
        self.config_path = Path(llm_config)
        self.model_name = "unknown"
        self.model_config = {}
        if self.config_path.exists():
            try:
                self.model_config = json.loads(self.config_path.read_text(encoding="utf-8"))
                self.model_name = self.model_config.get("model_name", self.model_config.get("model", "unknown"))
            except:
                pass

        self.llm_client = build_llm_client(llm_config)
        if hasattr(self.llm_client, "model"):
            self.model_name = self.llm_client.model
        elif hasattr(self.llm_client, "model_name"):
            self.model_name = self.llm_client.model_name

        self.prompt_repo = PromptRepository()
        self.predictor = Predictor(self.llm_client, self.prompt_repo)

    def run(self, input_path: Path, references_file: Optional[Path], output: Optional[Path], 
            recursive: bool, mode: str, extract_prompt_version: str, prompt_version: str, fill_empty_invariants: bool):
        
        references = load_references(references_file)
        command = " ".join(sys.argv)
        pmt_ver = prompt_version
        safe_pmt_ver = re.sub(r"[^\w\-]", "", pmt_ver)

        def _check_yaml_required_keys(path: Path, content: Any, strict: bool) -> bool:
            missing = validate_yaml_required_keys(path, content)
            if not missing:
                return True
            console.print(f"[red]YAML missing keys in {path}: {', '.join(missing)}[/red]")
            if strict:
                # We raise typer.Exit here as in original CLI, but maybe raising Exception is cleaner
                raise typer.Exit(code=1)
            return False

        def process_file(f: Path, base_dir: Optional[Path], output_root: Optional[Path], strict: bool):
            loops_to_analyze = []
            source_type = "code"
            source_path = str(f.relative_to(base_dir)) if base_dir else str(f)
            existing_yaml = None
            needs_fill = False
            input_has_extract = None
            input_source_file = None
            
            # Determine input type
            if f.suffix.lower() in {'.yml', '.yaml'}:
                source_type = "yaml"
                try:
                    data = yaml.safe_load(f.read_text(encoding="utf-8"))
                    if not _check_yaml_required_keys(f, data, strict):
                        return False

                    if isinstance(data, dict):
                        existing_yaml = data
                        if "source_path" in data:
                            source_path = data["source_path"]
                        if "source_file" in data:
                            input_source_file = data["source_file"]
                        if "has_extract" in data:
                            input_has_extract = data["has_extract"]

                    if isinstance(data, dict) and "invariants_result" in data:
                        # Re-infer or fill
                        if fill_empty_invariants:
                            needs_fill = True
                            # We only analyze empty ones
                            # But wait, original code iteration logic for 'loops_to_analyze' was slightly implicit in fill mode?
                            # Re-checking logic: Original code populates loops_to_analyze from data["invariants_result"] or data["loops"]
                            # BUT then if needs_fill is True, it iterates loops_to_analyze AND existing_yaml['invariants_result'] to match.
                            # So we must populate loops_to_analyze with ALL loops first.
                            pass
                        
                        loops_to_analyze = [item.get("code", "") for item in data["invariants_result"]]
                    elif "loops" in data:
                        loops_to_analyze = [item["code"] for item in data["loops"]]
                    else:
                        console.print(f"[yellow]YAML {f} format not recognized (expected 'loops' or 'invariants_result').[/yellow]")
                        return False
                        
                except Exception as e:
                    console.print(f"[red]Error parsing YAML {f}: {e}[/red]")
                    return False
            else:
                # Treat as raw code file
                code = f.read_text(encoding="utf-8")
                loops_to_analyze = [code]

            def infer_has_extract() -> bool:
                if source_type != "yaml":
                    return False
                if isinstance(input_has_extract, bool):
                    return input_has_extract
                if isinstance(input_source_file, str):
                    lower_name = input_source_file.lower()
                    if lower_name.endswith((".c", ".cpp", ".h", ".hpp", ".cc", ".cxx")):
                        return False
                    if lower_name.endswith((".yml", ".yaml")):
                        return True
                return True

            all_invariants = []
            for i, loop_code in enumerate(loops_to_analyze):
                # Check if we should skip this loop (for fill mode)
                if needs_fill and existing_yaml:
                    # Original logic was: iterate main loop, then later constructing result merging existing.
                    # But wait, to save LLM calls, we should only call infer if needed.
                    # The original code:
                    # for i, loop_code in enumerate(loops_to_analyze):
                    #    invariants = predictor.infer(...)
                    # It ran inference for ALL loops.
                    # Wait, let me check the original 'invariant' command logic for fill_empty_invariants.
                    # Ah, I don't see skipping logic in the loops. The logic for merge was AFTER:
                    # "if existing_yaml is not None and needs_fill: ... check if invs empty ... else use existing"
                    # This means it WAS re-inferring everything but only USING new ones if old were empty?
                    # That is wasteful. But if strict adherence to existing logic is required, I follow it.
                    # However, optimizing is "extracting logic". The user asked to clean/organize.
                    # If I see obvious waste, I can optimize, but risky if the logic was intentional (e.g. comparing).
                    # 'fill_empty_invariants' implies we only want to fill.
                    # The original code:
                    # for i, loop_code in enumerate(loops_to_analyze): invariants = predictor.infer(...)
                    # It seems it runs for all. Let's keep it 1:1 for now to ensure correctness of refactor.
                    pass

                invariants = self.predictor.infer_invariants(loop_code, references, prompt_version=prompt_version)
                all_invariants.append({
                    "loop_id": i + 1,
                    "code": loop_code,
                    "invariants": invariants
                })
                console.print(f"[green]Inferred {len(invariants)} invariants for loop {i+1} in {f.name}[/green]")

            # Output generation
            timestamp = datetime.now().strftime("%Y-%m-%dT%H:%M")
            has_extract = infer_has_extract()

            final_invariants_result = []

            if existing_yaml is not None and needs_fill:
                # Merge logic
                # For each existing item, if empty, take from new inference. 
                # Note: 'all_invariants' corresponds by index to 'loops_to_analyze'.
                # 'loops_to_analyze' came from the yaml list order.
                source_list = existing_yaml.get("invariants_result", [])
                
                # Create a map or just iterate by index if lengths match
                if len(source_list) != len(all_invariants):
                    console.print("[yellow]Warning: Mismatch in loop counts during fill merge. Fallback to full overwrite.[/yellow]")
                    final_invariants_result = all_invariants
                else:
                    for idx, item in enumerate(source_list):
                        existing_invs = item.get("invariants", [])
                        if not existing_invs: # Empty or None
                             # Use new
                             item["invariants"] = all_invariants[idx]["invariants"]
                             # also ensure code/id is preserved or updated? Keep existing structure.
                        final_invariants_result.append(item)
            else:
                final_invariants_result = all_invariants

            yaml_data = {
                "source_file": f.name,
                "source_path": source_path,
                "task": "invariant_inference",
                "command": command,
                "pmt_ver": pmt_ver,
                "model": str(self.model_name),
                "time": timestamp,
                "has_extract": has_extract,
                "invariants_result": final_invariants_result,
            }

            if existing_yaml is not None:
                for key, value in existing_yaml.items():
                    if key not in yaml_data and key != "basic":
                        yaml_data[key] = value

            base_name = f.stem
            if isinstance(source_path, str) and source_path.strip():
                base_name = Path(source_path).stem
            filename = f"{base_name}_{safe_pmt_ver}_inv.yml"
            
            if existing_yaml is not None and needs_fill:
                if output_root:
                    if output_root.suffix in {".yml", ".yaml"}:
                         out_path = output_root
                    else:
                         rel_parent = Path(".") if base_dir is None else f.parent.relative_to(base_dir)
                         result_dir = output_root / rel_parent
                         result_dir.mkdir(parents=True, exist_ok=True)
                         out_path = result_dir / filename
                else:
                    out_path = f
            else:
                if output_root:
                    rel_parent = Path(".") if base_dir is None else f.parent.relative_to(base_dir)
                    result_dir = output_root / rel_parent
                    result_dir.mkdir(parents=True, exist_ok=True)
                    out_path = result_dir / filename
                else:
                    result_dir = f.parent / "invariant_result"
                    result_dir.mkdir(exist_ok=True)
                    out_path = result_dir / filename
            
            with open(out_path, 'w', encoding='utf-8') as yf:
                yaml.dump(yaml_data, yf, Dumper=LiteralDumper, sort_keys=False, allow_unicode=True)
                
            console.print(f"Saved invariants to {out_path}")
            return True

        if input_path.is_file():
            output_root = None
            if output and output.suffix not in {".yml", ".yaml"}:
                if not output.exists():
                    output.mkdir(parents=True)
                output_root = output
            ok = process_file(input_path, None, output_root, True)
            if not ok:
                raise typer.Exit(code=1)
        
        elif input_path.is_dir():
            def matches_yaml_prompt_version(path: Path) -> bool:
                if not extract_prompt_version or extract_prompt_version.lower() in {"all", "auto"}:
                    return True
                version = extract_prompt_version.lower().lstrip("v")
                if version in {"1", "2"}:
                    token = f"pmt_yamlv{version}"
                    return token in path.name
                return True

            code_extensions = {".c", ".cpp", ".h", ".hpp", ".cc", ".cxx"}
            yaml_extensions = {".yml", ".yaml"}
            
            files = collect_files(input_path, recursive)
            
            filtered_files = []
            for f in files:
                ext = f.suffix.lower()
                if mode == "code":
                    if ext in code_extensions:
                        filtered_files.append(f)
                elif mode == "yaml":
                    if ext in yaml_extensions:
                        if matches_yaml_prompt_version(f):
                            filtered_files.append(f)
                else: # auto
                    if ext in code_extensions:
                        filtered_files.append(f)
                    elif ext in yaml_extensions:
                        if matches_yaml_prompt_version(f):
                            filtered_files.append(f)
            
            console.print(f"Found {len(filtered_files)} files to analyze (mode={mode}).")
            if output and not output.exists():
                output.mkdir(parents=True)
            for f in filtered_files:
                try:
                    process_file(f, input_path, output, False)
                except Exception as e:
                    console.print(f"[red]Error processing {f.name}: {e}[/red]")
