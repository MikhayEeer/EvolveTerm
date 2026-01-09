"""Handler for the 'extract' command."""
from __future__ import annotations
import json
import yaml
import sys
import re
from pathlib import Path
from typing import Optional
from datetime import datetime
from rich.console import Console

from ..loop_extractor import LoopExtractor
from ..prompts_loader import PromptRepository
from ..llm_client import build_llm_client
from ..cli_utils import collect_files
from ..utils import LiteralDumper

console = Console()

class ExtractHandler:
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
        self.extractor = LoopExtractor(self.llm_client, self.prompt_repo)

    def run(self, input_path: Path, output: Optional[Path], recursive: bool, prompt_version: str):
        prompt_name = f"loop_extraction/yaml_{prompt_version}"
        pmt_ver = f"pmt_yaml{prompt_version}"
        safe_pmt_ver = re.sub(r"[^\w\-]", "", pmt_ver)
        command = " ".join(sys.argv)

        def process_file(f: Path, base_dir: Optional[Path], output_root: Optional[Path]):
            code = f.read_text(encoding="utf-8")
            loops = self.extractor.extract(code, prompt_name=prompt_name)
            
            timestamp = datetime.now().strftime("%Y-%m-%dT%H:%M")
            yaml_data = {
                "source_path": str(f.relative_to(base_dir)) if base_dir else str(f),
                "task": "extract",
                "command": command,
                "pmt_ver": pmt_ver,
                "model": str(self.model_name),
                "time": timestamp,
                "loops_count": -1,
                "loops_depth": -1,
                "loops_ids": len(loops),
                "loops": [
                    {
                        "id": i + 1,
                        "code": loop.replace('\t', '    ')
                    }
                    for i, loop in enumerate(loops)
                ]
            }
            
            filename = f"{f.stem}_{safe_pmt_ver}_extract.yml"

            if output_root:
                if output_root.suffix.lower() in {'.yml', '.yaml'} and not base_dir:
                    out_path = output_root
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                else:
                    if base_dir:
                        rel_path = f.parent.relative_to(base_dir)
                        result_dir = output_root / rel_path
                    else:
                        result_dir = output_root
                    
                    result_dir.mkdir(parents=True, exist_ok=True)
                    out_path = result_dir / filename
            else:
                # Output directory: sibling 'extract_result'
                result_dir = f.parent / "extract_result"
                result_dir.mkdir(exist_ok=True)
                out_path = result_dir / filename
            
            with open(out_path, 'w', encoding='utf-8') as yf:
                yaml.dump(yaml_data, yf, Dumper=LiteralDumper, sort_keys=False, allow_unicode=True)
                
            console.print(f"Saved extraction to {out_path}")

        if input_path.is_file():
            console.print(f"Extracting loops from {input_path}...")
            process_file(input_path, None, output)
                
        elif input_path.is_dir():
            extensions = {".c", ".cpp", ".h", ".hpp", ".cc", ".cxx"}
            files = collect_files(input_path, recursive, extensions=extensions)
            console.print(f"[bright_cyan]INFO:[/bright_cyan] Found {len(files)} files.")
            
            for f in files:
                try:
                    process_file(f, input_path, output)
                except Exception as e:
                    console.print(f"[bold red]ERROR: Error extracting {f.name}: {e}[/bold red]")
