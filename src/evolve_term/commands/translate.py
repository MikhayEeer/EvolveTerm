from pathlib import Path
from typing import Optional
from rich.console import Console
from evolve_term.translator import CodeTranslator
from evolve_term.cli_utils import collect_files, ensure_output_dir

console = Console()

class TranslateHandler:
    def __init__(self, llm_config: str):
        self.translator = CodeTranslator(config_name=llm_config)

    def run(
        self,
        input: Path,
        output: Optional[Path],
        recursive: bool
    ) -> None:
        if input.is_file():
            console.print(f"Translating {input}...")
            code = input.read_text(encoding="utf-8")
            translated = self.translator.translate(code)
            
            if output:
                if output.is_dir():
                    out_path = output / (input.stem + ".cpp")
                else:
                    out_path = output
                out_path.write_text(translated, encoding="utf-8")
                console.print(f"Saved to {out_path}")
            else:
                console.print(translated)
                
        elif input.is_dir():
            files = collect_files(input, recursive)
            console.print(f"[bright_cyan]INFO:[/bright_cyan] Found {len(files)} files to translate.")
            
            ensure_output_dir(output)
                
            for f in files:
                try:
                    console.print(f"Translating {f.name}...")
                    code = f.read_text(encoding="utf-8")
                    translated = self.translator.translate(code)
                    
                    if output:
                        # Maintain relative path structure if possible, else flat
                        try:
                            rel_path = f.relative_to(input)
                            out_path = output / rel_path.with_suffix(".cpp")
                        except ValueError:
                            out_path = output / (f.stem + ".cpp")
                        
                        out_path.parent.mkdir(parents=True, exist_ok=True)
                        out_path.write_text(translated, encoding="utf-8")
                    else:
                        console.print(f"--- {f.name} ---")
                        console.print(translated)
                except Exception as e:
                    console.print(f"[bold red]ERROR: Error translating {f.name}: {e}[/bold red]")
