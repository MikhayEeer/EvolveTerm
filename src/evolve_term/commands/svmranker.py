"""Handler for the 'svmranker' command."""
from __future__ import annotations
import json
import yaml
import tempfile
from pathlib import Path
from typing import Optional, List, Dict, Any
from rich.console import Console
import typer

from ..svm_ranker import SVMRankerClient
from ..cli_utils import collect_files, validate_yaml_required_keys, ensure_output_dir
from ..utils import LiteralDumper

console = Console()

def resolve_svm_ranker_root(path: Path) -> Path:
    if not path.exists():
        raise typer.BadParameter(f"SVMRanker 路径不存在: {path}")
    if path.is_file():
        if path.name == "CLIMain.py" and path.parent.name == "src":
            path = path.parent.parent
        else:
            raise typer.BadParameter("SVMRanker 路径应为仓库根目录或 src/CLIMain.py 文件。")

    if (path / "src" / "CLIMain.py").exists():
        return path
    if path.name == "src" and (path / "CLIMain.py").exists():
        return path.parent
    raise typer.BadParameter("SVMRanker 路径无效，未找到 src/CLIMain.py。")

class SVMRankerHandler:
    def __init__(self, svm_ranker_path: Path):
        self.root = resolve_svm_ranker_root(svm_ranker_path)
        self.client = SVMRankerClient(str(self.root))

    def run(self, input_path: Path, output: Optional[Path], recursive: bool):
        
        def _check_yaml_required_keys(path: Path, content: Any, strict: bool) -> bool:
            missing = validate_yaml_required_keys(path, content)
            if not missing:
                return True
            console.print(f"[red]YAML missing keys in {path}: {', '.join(missing)}[/red]")
            if strict:
                return False
            return False

        def parse_results(content: Any) -> List[Dict[str, Any]]:
            if isinstance(content, dict) and "ranking_results" in content:
                return content["ranking_results"] or []
            if isinstance(content, list):
                return content
            return []

        def resolve_source_code(entry: Dict[str, Any], base_dir: Path) -> str:
            source_path = entry.get("source_path")
            if source_path:
                path = Path(source_path)
                if not path.is_absolute():
                     path = base_dir / path
                if path.exists():
                    try:
                        return path.read_text(encoding="utf-8")
                    except Exception:
                        pass
                    
            return entry.get("code", "")

        def run_on_entry(entry: Dict[str, Any], base_dir: Path) -> Dict[str, Any]:
            code = resolve_source_code(entry, base_dir)
            template_type = entry.get("template_type") or entry.get("type") or "lnested"
            template_depth = entry.get("template_depth") or entry.get("depth") or 1
            mode = "lmulti" if "multi" in str(template_type).lower() else "lnested"
            try:
                depth_val = int(template_depth)
            except Exception:
                depth_val = 1

            with tempfile.NamedTemporaryFile(mode="w", suffix=".c", delete=False, encoding="utf-8") as tmp:
                tmp.write(code)
                tmp_path = tmp.name

            try:
                status, rf, rf_list = self.client.run(Path(tmp_path), mode=mode, depth=depth_val)
            finally:
                Path(tmp_path).unlink(missing_ok=True)

            return {
                "loop_id": entry.get("loop_id") or entry.get("id"),
                "template_type": template_type,
                "template_depth": depth_val,
                "svm_mode": mode,
                "status": status,
                "ranking_function": rf,
                "ranking_functions": rf_list,
            }

        def process_yaml(f: Path, strict: bool) -> Optional[Dict[str, Any]]:
            try:
                content = yaml.safe_load(f.read_text(encoding="utf-8"))
            except Exception as e:
                return {"source_file": f.name, "error": f"YAML parse error: {e}"}
            if not _check_yaml_required_keys(f, content, strict):
                return None

            entries = parse_results(content)
            base_dir = f.parent
            results = [run_on_entry(entry, base_dir) for entry in entries]
            return {"source_file": f.name, "task": "svmranker", "results": results}

        if input_path.is_file():
            result = process_yaml(input_path, True)
            if result is None:
                raise typer.Exit(code=1)
            
            if output:
                if output.is_dir():
                    out_path = output / (input_path.stem + "_svm.yml")
                else:
                    out_path = output
                
                with open(out_path, "w", encoding="utf-8") as yf:
                    yaml.dump(result, yf, sort_keys=False, allow_unicode=True)
                console.print(f"Saved to {out_path}")
            else:
                console.print(yaml.dump(result, sort_keys=False, allow_unicode=True))

        elif input_path.is_dir():
            files = collect_files(input_path, recursive, extensions={".yml", ".yaml"})

            ensure_output_dir(output)

            for f in files:
                try:
                    result = process_yaml(f, False)
                    if not result: continue
                    
                    if output:
                        try:
                             rel_path = f.relative_to(input_path)
                             out_path = output / rel_path.with_suffix(".yml")
                             out_path = out_path.parent / (out_path.stem + "_svm.yml")
                        except ValueError:
                             out_path = output / (f.stem + "_svm.yml")
                        
                        out_path.parent.mkdir(parents=True, exist_ok=True)
                        with open(out_path, "w", encoding="utf-8") as yf:
                             yaml.dump(result, yf, sort_keys=False, allow_unicode=True)
                    else:
                        console.print(f"--- {f.name} ---")
                        console.print(yaml.dump(result, sort_keys=False, allow_unicode=True))
                        
                except Exception as e:
                    console.print(f"[red]Error processing {f.name}: {e}[/red]")
