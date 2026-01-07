"""Handler for the 'predict' command."""
from __future__ import annotations
import json
import yaml
from pathlib import Path
from typing import Optional, List, Dict, Any
from rich.console import Console
import typer

from ..predict import Predictor
from ..loop_extractor import LoopExtractor
from ..prompts_loader import PromptRepository
from ..llm_client import build_llm_client
from ..cli_utils import collect_files, load_references, validate_yaml_required_keys, ensure_output_dir
from ..utils import LiteralDumper

console = Console()

class PredictHandler:
    def __init__(self, llm_config: str):
        self.llm_client = build_llm_client(llm_config)
        self.prompt_repo = PromptRepository()
        self.predictor = Predictor(self.llm_client, self.prompt_repo)
        self.loop_extractor = LoopExtractor(self.llm_client, self.prompt_repo)

    def run(self, input_path: Path, references_file: Optional[Path], output: Optional[Path], recursive: bool):
        references = load_references(references_file)

        def _check_yaml_required_keys(path: Path, content: Any, strict: bool) -> bool:
            missing = validate_yaml_required_keys(path, content)
            if not missing:
                return True
            console.print(f"[red]YAML missing keys in {path}: {', '.join(missing)}[/red]")
            if strict:
                return False
            return False

        def process_yaml_input(f: Path, strict: bool) -> Optional[List[Dict[str, Any]]]:
            try:
                content = yaml.safe_load(f.read_text(encoding="utf-8"))
            except Exception as e:
                console.print(f"[red]Error parsing YAML {f}: {e}[/red]")
                return None

            if not _check_yaml_required_keys(f, content, strict):
                return None

            results = []
            
            # Case 1: Invariant Result YAML
            if "invariants_result" in content:
                console.print(f"[blue]Detected Invariant Result YAML: {f.name}[/blue]")
                for item in content["invariants_result"]:
                    loop_id = item.get("loop_id") or item.get("id")
                    code = item.get("code", "")
                    invs = item.get("invariants", [])
                    
                    console.print(f"  Predicting for Loop {loop_id}...")
                    # We pass the loop code as 'code', and also as the single item in 'loops' list
                    prediction, _ = self.predictor.predict(code, [code], references, invariants=invs)
                    
                    results.append({
                        "loop_id": loop_id,
                        "prediction": prediction
                    })

            # Case 2: Extract Result YAML
            elif "loops" in content:
                console.print(f"[blue]Detected Extract Result YAML: {f.name}[/blue]")
                for item in content["loops"]:
                    loop_id = item.get("id")
                    code = item.get("code", "")
                    
                    console.print(f"  Predicting for Loop {loop_id}...")
                    prediction, _ = self.predictor.predict(code, [code], references)
                    results.append({
                        "loop_id": loop_id,
                        "prediction": prediction
                    })
                    
            else:
                console.print(f"[yellow]Unknown YAML format in {f.name}.[/yellow]")
                
            return results

        def process_c_file(f: Path) -> Dict[str, Any]:
            code = f.read_text(encoding="utf-8")
            loops = self.loop_extractor.extract(code)
            prediction, _ = self.predictor.predict(code, loops, references)
            return {
                "file": f.name,
                "prediction": prediction
            }

        if input_path.is_file():
            if input_path.suffix.lower() in {'.yml', '.yaml'}:
                results = process_yaml_input(input_path, True)
                if results is None:
                    raise typer.Exit(code=1)
                
                if output:
                    if output.is_dir():
                        out_path = output / (input_path.stem + "_pred.json")
                    else:
                        out_path = output
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
                    console.print(f"Saved to {out_path}")
                else:
                    console.print(json.dumps(results, indent=2))
                    
            else:
                console.print(f"Predicting for {input_path}...")
                result = process_c_file(input_path)
                if output:
                    if output.is_dir():
                        out_path = output / (input_path.stem + ".json")
                    else:
                        out_path = output
                    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
                    console.print(f"Saved to {out_path}")
                else:
                    console.print(json.dumps(result, indent=2))

        elif input_path.is_dir():
            files = collect_files(input_path, recursive)
            
            ensure_output_dir(output)
                
            for f in files:
                try:
                    if f.suffix.lower() in {'.yml', '.yaml'}:
                         results = process_yaml_input(f, False)
                         if results:
                             if output:
                                try:
                                    rel_path = f.relative_to(input_path)
                                    out_path = output / rel_path.with_suffix(".json")
                                    # Modify stem
                                    out_path = out_path.parent / (out_path.stem + "_pred.json")
                                except ValueError:
                                    out_path = output / (f.stem + "_pred.json")
                                out_path.parent.mkdir(parents=True, exist_ok=True)
                                out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
                             else:
                                 console.print(f"--- {f.name} ---")
                                 console.print(json.dumps(results, indent=2))

                    elif f.suffix.lower() in {'.c', '.cpp', '.h'}: # simplified extension check
                        console.print(f"Predicting for {f.name}...")
                        result = process_c_file(f)
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
