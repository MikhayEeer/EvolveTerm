from pathlib import Path
from typing import Optional, List
import json
import yaml
import typer
from rich.console import Console

from evolve_term.llm_client import build_llm_client
from evolve_term.cli_utils import collect_files, ensure_output_dir, load_json_or_yaml
from evolve_term.prompts_loader import PromptRepository
from evolve_term.verifier import Z3Verifier

console = Console()

class Z3VerifyHandler:
    def __init__(self, llm_config: str):
        self.llm_client = build_llm_client(llm_config)
        self.prompt_repo = PromptRepository()
        self.verifier = Z3Verifier(self.llm_client, self.prompt_repo)

    def run(
        self,
        input: Path,
        ranking_func: Optional[str],
        ranking_file: Optional[Path],
        invariants_file: Optional[Path],
        output: Optional[Path],
        recursive: bool,
    ) -> None:
        
        def get_rf(f_path: Path) -> Optional[str]:
            # If explicit string provided, use it (only valid for single file really)
            if ranking_func:
                return ranking_func
            
            # If ranking file provided
            if ranking_file:
                # If ranking_file is a directory, try to find corresponding file
                if ranking_file.is_dir():
                    # Try to find file with same stem (JSON or YAML)
                    for ext in [".json", ".yml", ".yaml", ".txt"]:
                        candidate = ranking_file / (f_path.stem + ext)
                        if candidate.exists():
                            content = candidate.read_text(encoding="utf-8")
                            try:
                                data = load_json_or_yaml(candidate)
                                if isinstance(data, dict):
                                    return data.get("ranking_function")
                                return str(data)
                            except:
                                return content.strip()
                    return None
                else:
                    # Single ranking file provided
                    if ranking_file.exists():
                        content = ranking_file.read_text(encoding="utf-8")
                        try:
                            data = load_json_or_yaml(ranking_file)
                            if isinstance(data, dict):
                                return data.get("ranking_function")
                            return str(data)
                        except:
                            return content.strip()
            return None

        def get_invs(f_path: Path) -> List[str]:
            if invariants_file:
                if invariants_file.is_dir():
                    # Try both JSON and YAML extensions
                    for ext in [".json", ".yml", ".yaml"]:
                        candidate = invariants_file / (f_path.stem + ext)
                        if candidate.exists():
                            try:
                                data = load_json_or_yaml(candidate)
                                if isinstance(data, list):
                                    return data
                                elif isinstance(data, dict) and "invariants" in data:
                                    return data["invariants"]
                                return []
                            except:
                                return []
                elif invariants_file.exists():
                    try:
                        data = load_json_or_yaml(invariants_file)
                        if isinstance(data, list):
                            return data
                        elif isinstance(data, dict) and "invariants" in data:
                            return data["invariants"]
                        return []
                    except:
                        return []
            return []

        def process_file(f: Path) -> str:
            code = f.read_text(encoding="utf-8")
            rf = get_rf(f)
            invs = get_invs(f)
            
            if not rf:
                return "Skipped (No Ranking Function)"
                
            return self.verifier.verify(code, invs, rf)

        if input.is_file():
            console.print(f"Verifying {input}...")
            result = process_file(input)
            
            if output:
                if output.is_dir():
                    out_path = output / (input.stem + ".txt")
                else:
                    out_path = output
                # ensure parent exists
                if not out_path.parent.exists():
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_text(result, encoding="utf-8")
                console.print(f"Saved to {out_path}")
            else:
                console.print(f"Result: {result}")
                
        elif input.is_dir():
            files = collect_files(input, recursive)
            
            ensure_output_dir(output)
                
            for f in files:
                try:
                    console.print(f"Verifying {f.name}...")
                    result = process_file(f)
                    
                    if output:
                        try:
                            rel_path = f.relative_to(input)
                            out_path = output / rel_path.with_suffix(".txt")
                        except ValueError:
                            out_path = output / (f.stem + ".txt")
                        
                        out_path.parent.mkdir(parents=True, exist_ok=True)
                        out_path.write_text(result, encoding="utf-8")
                    else:
                        console.print(f"--- {f.name} ---")
                        console.print(f"Result: {result}")
                except Exception as e:
                    console.print(f"[bold red]ERROR: Error verifying {f.name}: {e}[/bold red]")
