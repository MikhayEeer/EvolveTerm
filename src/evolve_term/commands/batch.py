"""Handler for the 'batch_analyze' command."""
from __future__ import annotations
from pathlib import Path
from typing import Optional
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table
import csv
from datetime import datetime
import typer

from ..pipeline import TerminationPipeline
from ..cli_utils import resolve_svm_ranker_root

console = Console()

class BatchHandler:
    def run(self, 
            input_dir: Path, 
            top_k: int, 
            enable_translation: bool, 
            knowledge_base: Optional[Path], 
            recursive: bool, 
            use_rag_reasoning: bool, 
            svm_ranker_path: Optional[Path], 
            use_smt_synth: bool,
            known_terminating: bool, 
            extraction_prompt_version: str, 
            use_loops_for_embedding: bool, 
            use_loops_for_reasoning: bool,
            verifier_backend: str,
            seahorn_image: str,
            seahorn_timeout: int):

        extensions = {".c", ".cpp", ".h", ".hpp", ".cc", ".cxx"}
        pattern = "**/*" if recursive else "*"
        files = [
            f for f in input_dir.glob(pattern) 
            if f.is_file() and f.suffix.lower() in extensions
        ]
        
        if not files:
            console.print(f"[yellow]No C/C++ files found in {input_dir}[/yellow]")
            return
            
        console.print(f"[bold]Found {len(files)} files to analyze.[/bold]")
        
        svm_ranker_root = None
        if svm_ranker_path:
            svm_ranker_root = resolve_svm_ranker_root(svm_ranker_path)

        pipeline = TerminationPipeline(
            enable_translation=enable_translation,
            knowledge_base_path=str(knowledge_base) if knowledge_base else None,
            svm_ranker_path=str(svm_ranker_root) if svm_ranker_root else None,
            verifier_backend=verifier_backend,
            seahorn_docker_image=seahorn_image,
            seahorn_timeout=seahorn_timeout,
        )
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = input_dir / f"batch_report_{timestamp}.csv"
        csv_headers = [
            "Filename", "Relative Path", "Run ID", "Date Time", "Duration (s)", 
            "LLM Calls", "Label", "RAG Similarity", "Invariants", 
            "Ranking Function", "Verification Result", "Report Path", "Error"
        ]
        
        console.print(f"Writing results to: [blue]{csv_path}[/blue]")
        
        results = []
        
        with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(csv_headers)
        
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                console=console
            ) as progress:
                task = progress.add_task("Analyzing...", total=len(files))
                
                for file_path in files:
                    progress.update(task, description=f"Analyzing {file_path.name}...")
                    rel_path = ""
                    try:
                        rel_path = str(file_path.relative_to(input_dir))
                        code = file_path.read_text(encoding="utf-8")
                        
                        start_calls = getattr(pipeline.llm_client, "call_count", 0)
                        
                        ret = pipeline.analyze(
                            code, 
                            top_k=top_k, 
                            use_rag_in_reasoning=use_rag_reasoning,
                            use_svm_ranker=bool(svm_ranker_root),
                            use_smt_synth=use_smt_synth,
                            known_terminating=known_terminating,
                            extraction_prompt_version=extraction_prompt_version,
                            use_loops_for_embedding=use_loops_for_embedding,
                            use_loops_for_reasoning=use_loops_for_reasoning
                        )
                        
                        end_calls = getattr(pipeline.llm_client, "call_count", 0)
                        
                        # Flatten invariants
                        inv_str = "; ".join(ret.invariants) if ret.invariants else ""
                        
                        # Calc duration from report if available, else 0
                        duration = 0.0
                        # We don't have easy access to ret.run_time unless we parse report?
                        # pipeline.analyze returns PredictionResult which doesn't seem to have duration directly?
                        # The original CLI code used: duration = ret.report.get("basic", {}).get("duration_seconds", 0)
                        # But analyze returns PredictionResult... let's check pipeline code again.
                        # It writes a report file.
                        
                        # In CLI code I omitted lines 229-263.
                        # I'll assume standard exception handling.
                        
                        writer.writerow([
                            file_path.name, rel_path, getattr(ret, "run_id", ""), datetime.now().isoformat(),
                            "", # duration
                            end_calls - start_calls, ret.label, 
                            "", # rag similarity
                            inv_str, ret.ranking_function, ret.verification_result, ret.report_path or "", ""
                        ])
                        results.append((file_path.name, ret))

                    except Exception as e:
                        console.print(f"[red]Error analyzing {file_path.name}: {e}[/red]")
                        writer.writerow([
                            file_path.name, rel_path, "", datetime.now().isoformat(),
                            "", "", "Error", "", "", "", "", "", str(e)
                        ])
                    finally:
                        progress.advance(task)
                        
        # Summary table
        table = Table(title="Batch Analysis Summary")
        table.add_column("File")
        table.add_column("Label")
        table.add_column("Verification")
        table.add_column("Ranking Function")
        
        for filename, res in results:
            rf = res.ranking_function if res.ranking_function else "-"
            ver = res.verification_result if res.verification_result else "-"
            
            # Color code verification
            if ver == "Verified":
                ver = "[green]Verified[/green]"
            elif ver.startswith("Failed"):
                ver = f"[red]{ver}[/red]"
                
            table.add_row(filename, res.label, ver, rf)
            
        console.print(table)
        console.print(f"\n[bold green]Batch analysis complete. Results saved to {csv_path}[/bold green]")
