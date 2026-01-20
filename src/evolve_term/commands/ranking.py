"""Handler for the 'ranking' command."""
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
from ..cli_utils import collect_files, load_references, validate_yaml_required_keys, ensure_output_dir
from ..utils import LiteralDumper

console = Console()

class RankingHandler:
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

    def run(self, input_path: Path, invariants_file: Optional[Path], references_file: Optional[Path], 
            output: Optional[Path], recursive: bool, mode: str, ranking_mode: str, retry_empty: int):
        
        ranking_mode_value = ranking_mode.strip()
        ranking_mode_key = ranking_mode_value.lower()
        rf_prompt_name = None

        if ranking_mode_key in {"template-known", "template_known"}:
            rf_mode = "template"
            rf_known_terminating = True
            pmt_ver = "template-known"
        elif ranking_mode_key in {"template-known-fewshot", "template_known_fewshot"}:
            rf_mode = "template_fewshot"
            rf_known_terminating = True
            pmt_ver = "template-known-fewshot"
        elif ranking_mode_key in {"template-fewshot", "template_fewshot"}:
            rf_mode = "template_fewshot"
            rf_known_terminating = False
            pmt_ver = "template-fewshot"
        elif ranking_mode_key in {"direct", "template"}:
            rf_mode = ranking_mode_key
            rf_known_terminating = False
            pmt_ver = ranking_mode_key
        else:
            prompt_base = ranking_mode_key
            if prompt_base.startswith("ranking_function/"):
                prompt_base = prompt_base.split("/", 1)[1]
            prompt_dir = self.prompt_repo.root / "ranking_function"
            system_path = prompt_dir / f"{prompt_base}.system.txt"
            user_path = prompt_dir / f"{prompt_base}.user.txt"
            if system_path.exists() and user_path.exists():
                rf_prompt_name = f"ranking_function/{prompt_base}"
                pmt_ver = prompt_base
                if prompt_base.startswith("rf_template") or "template" in prompt_base:
                    rf_mode = "template"
                else:
                    rf_mode = "direct"
                rf_known_terminating = "known" in prompt_base
            else:
                raise typer.BadParameter(
                    "ranking_mode must be one of: direct, template, template-fewshot, "
                    "template-known, template-known-fewshot, or a prompt name under "
                    "prompts/ranking_function (e.g. rf_direct_simple_known)"
                )

        safe_pmt_ver = re.sub(r"[^\w\-]", "", pmt_ver)
        command = " ".join(sys.argv)
        
        references = load_references(references_file)
        retry_empty = max(0, retry_empty)

        def is_empty_result(rf: str | None, metadata: dict) -> bool:
            return self.predictor.is_empty_ranking_result(rf, metadata, rf_mode)

        def is_template_mode() -> bool:
            return isinstance(rf_mode, str) and rf_mode.startswith("template")

        def derive_base_name(source_path: Optional[str], fallback: Path) -> str:
            if isinstance(source_path, str) and source_path.strip():
                return Path(source_path).stem
            return fallback.stem

        def has_non_empty_invariants(results: List[Dict[str, Any]]) -> bool:
            for item in results:
                if not isinstance(item, dict):
                    continue
                invs = item.get("invariants")
                if isinstance(invs, list):
                    if any(str(val).strip() for val in invs if val is not None):
                        return True
                elif isinstance(invs, str) and invs.strip():
                    return True
            return False

        def _check_yaml_required_keys(path: Path, content: Any, strict: bool) -> bool:
            missing = validate_yaml_required_keys(path, content)
            if not missing:
                return True
            console.print(f"[red]YAML missing keys in {path}: {', '.join(missing)}[/red]")
            if strict:
                return False
            return False

        def process_file_code(f: Path, invs: List[str]) -> Dict[str, Any]:
            code = f.read_text(encoding="utf-8")
            rf, explanation, metadata = self.predictor.infer_ranking(
                code,
                invs,
                references,
                mode=rf_mode,
                known_terminating=rf_known_terminating,
                retry_empty=retry_empty,
                log_prefix=f.name,
                prompt_name=rf_prompt_name,
            )
            if is_empty_result(rf, metadata):
                return {"status": "empty", "explanation": explanation}
            if is_template_mode():
                return {
                    "template_type": metadata.get("type"),
                    "template_depth": metadata.get("depth"),
                    "explanation": explanation,
                }
            return {"ranking_function": rf, "explanation": explanation}

        def process_yaml_input(f: Path, strict: bool) -> tuple[Optional[List[Dict[str, Any]]], str, bool, bool]:
            try:
                content = yaml.safe_load(f.read_text(encoding="utf-8"))
            except Exception as e:
                console.print(f"[red]Error parsing YAML {f}: {e}[/red]")
                return None, str(f), False, False

            if not _check_yaml_required_keys(f, content, strict):
                return None, str(f), False, False

            results = []
            source_path = content.get("source_path") if isinstance(content, dict) else None
            has_extract = False
            if isinstance(content, dict):
                if isinstance(content.get("has_extract"), bool):
                    has_extract = content["has_extract"]
                elif "loops" in content or "invariants_result" in content:
                    has_extract = True
            
            # Case 1: Invariant Result YAML
            if isinstance(content, dict) and "invariants_result" in content:
                console.print(f"[blue]Detected Invariant Result YAML: {f.name}[/blue]")
                for item in content["invariants_result"]:
                    loop_id = item.get("loop_id") or item.get("id")
                    code = item.get("code", "")
                    invs = item.get("invariants", [])
                    
                    console.print(f"  Inferring ranking for Loop {loop_id}...")
                    rf, explanation, metadata = self.predictor.infer_ranking(
                        code,
                        invs,
                        references,
                        mode=rf_mode,
                        known_terminating=rf_known_terminating,
                        retry_empty=retry_empty,
                        log_prefix=f"{f.name} loop {loop_id}",
                        prompt_name=rf_prompt_name,
                    )
                    
                    result_entry = {
                        "loop_id": loop_id,
                        "code": code,
                        "invariants": invs,
                        "explanation": explanation,
                    }
                    if is_empty_result(rf, metadata):
                         result_entry["status"] = "empty"
                    elif is_template_mode():
                        result_entry["template_type"] = metadata.get("type")
                        result_entry["template_depth"] = metadata.get("depth")
                    else:
                        result_entry["ranking_function"] = rf
                    
                    results.append(result_entry)
                    
            # Case 2: Extract Result YAML
            elif isinstance(content, dict) and "loops" in content:
                console.print(f"[blue]Detected Extract Result YAML: {f.name}[/blue]")
                for item in content["loops"]:
                    loop_id = item.get("id")
                    code = item.get("code", "")
                    # No invariants in extract result
                    
                    console.print(f"  Inferring ranking for Loop {loop_id}...")
                    rf, explanation, metadata = self.predictor.infer_ranking(
                        code,
                        [],
                        references,
                        mode=rf_mode,
                        known_terminating=rf_known_terminating,
                        retry_empty=retry_empty,
                        log_prefix=f"{f.name} loop {loop_id}",
                        prompt_name=rf_prompt_name,
                    )
                    
                    result_entry = {
                        "loop_id": loop_id,
                        "code": code,
                        "invariants": [],
                        "explanation": explanation,
                    }
                    if is_empty_result(rf, metadata):
                        result_entry["status"] = "empty"
                    elif is_template_mode():
                        result_entry["template_type"] = metadata.get("type")
                        result_entry["template_depth"] = metadata.get("depth")
                    else:
                        result_entry["ranking_function"] = rf
                        
                    results.append(result_entry)
            else:
                console.print(f"[yellow]Unknown YAML format in {f.name}. Expected 'loops' or 'invariants_result'.[/yellow]")
                
            has_invariants = has_non_empty_invariants(results)
            return results, source_path or str(f), has_extract, has_invariants

        if input_path.is_file():
            if input_path.suffix.lower() in {'.yml', '.yaml'}:
                results, source_path, has_extract, has_invariants = process_yaml_input(input_path, True)
                if results is None:
                    raise typer.Exit(code=1)
                
                base_name = derive_base_name(source_path, input_path)
                timestamp = datetime.now().strftime("%Y-%m-%dT%H:%M")
                
                if output:
                    if output.is_dir():
                        out_path = output / f"{base_name}_{safe_pmt_ver}_ranking.yml"
                    else:
                        out_path = output
                    
                    # Wrap in a structure
                    out_data = {
                        "source_file": input_path.name,
                        "source_path": source_path,
                        "task": "ranking_inference",
                        "command": command,
                        "pmt_ver": pmt_ver,
                        "model": str(self.model_name),
                        "time": timestamp,
                        "has_extract": has_extract,
                        "has_invariants": has_invariants,
                        "ranking_results": results
                    }
                    
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(out_path, 'w', encoding='utf-8') as f:
                        yaml.dump(out_data, f, Dumper=LiteralDumper, sort_keys=False, allow_unicode=True)
                    console.print(f"Saved ranking results to {out_path}")
                else:
                    console.print(yaml.dump(results, sort_keys=False, allow_unicode=True))

            else:
                # Code file
                console.print(f"Inferring ranking function for {input_path}...")
                invariants = []
                if invariants_file:
                    try:
                         invariants = json.loads(invariants_file.read_text(encoding="utf-8"))
                    except Exception as e:
                         console.print(f"[red]Error reading invariants file: {e}[/red]")
                    
                result = process_file_code(input_path, invariants)
                
                if output:
                    if output.is_dir():
                        out_path = output / (input_path.stem + ".json")
                    else:
                        out_path = output
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
                    console.print(f"Saved to {out_path}")
                else:
                    console.print(json.dumps(result, indent=2))
                
        elif input_path.is_dir():
            if invariants_file:
                console.print("[yellow]Warning: --invariants-file ignored in batch mode.[/yellow]")
            
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
                        filtered_files.append(f)
                else: 
                     if ext in code_extensions or ext in yaml_extensions:
                         filtered_files.append(f)
            
            console.print(f"Found {len(filtered_files)} files to analyze (mode={mode}).")
            
            ensure_output_dir(output)
                
            for f in filtered_files:
                try:
                    console.print(f"Processing {f.name}...")
                    
                    if f.suffix.lower() in yaml_extensions:
                        results, source_path, has_extract, has_invariants = process_yaml_input(f, False)
                        if results is None:
                             continue

                        if output:
                            try:
                                rel_path = f.relative_to(input_path)
                                # Try to reconstruct output naming based on file name or relative path
                                # This logic in CLI was a bit duplicated.
                                out_path = output / rel_path.with_suffix(".yml")
                                out_path = out_path.parent / (out_path.stem + f"_{safe_pmt_ver}_ranking.yml")
                            except ValueError:
                                out_path = output / (f.stem + f"_{safe_pmt_ver}_ranking.yml")
                            
                            timestamp = datetime.now().strftime("%Y-%m-%dT%H:%M")
                            out_data = {
                                "source_file": f.name,
                                "source_path": source_path,
                                "task": "ranking_inference",
                                "command": command,
                                "pmt_ver": pmt_ver,
                                "model": str(self.model_name),
                                "time": timestamp,
                                "has_extract": has_extract,
                                "has_invariants": has_invariants,
                                "ranking_results": results
                            }

                            out_path.parent.mkdir(parents=True, exist_ok=True)
                            with open(out_path, 'w', encoding='utf-8') as yf:
                                yaml.dump(out_data, yf, Dumper=LiteralDumper, sort_keys=False, allow_unicode=True)
                        else:
                            console.print(f"--- {f.name} ---")
                            console.print(yaml.dump(results, sort_keys=False, allow_unicode=True))
                    else:
                        # C/C++ file
                        result = process_file_code(f, [])
                        if output:
                            try:
                                rel_path = f.relative_to(input_path)
                                out_path = output / rel_path.with_suffix(".json")
                            except ValueError:
                                out_path = output / (f.stem + ".json")
                            
                            out_path.parent.mkdir(parents=True, exist_ok=True)
                            out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
                        else:
                            console.print(f"--- {f.name} ---")
                            console.print(json.dumps(result, indent=2))
                            
                except Exception as e:
                    console.print(f"[red]Error processing {f.name}: {e}[/red]")
