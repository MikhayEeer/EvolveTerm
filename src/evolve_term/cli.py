"""Typer-based CLI to interact with the termination pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from .models import PendingReviewCase
from .pipeline import TerminationPipeline

app = typer.Typer(help="EvolveTerm CLI - analyze and curate C termination cases")
console = Console()
VALID_LABELS = {"terminating", "non-terminating", "unknown"}


@app.command()
def analyze(
    code_file: Path = typer.Option(..., exists=True, readable=True, help="Path to a source file"),
    top_k: int = 5,
    enable_translation: bool = typer.Option(False, "--enable-translation", "-t", help="Enable LLM-based translation to C++ for non-C/C++ files"),
    knowledge_base: Optional[Path] = typer.Option(None, "--kb", help="Path to a custom knowledge base JSON file"),
    use_rag_reasoning: bool = typer.Option(True, "--use-rag-reasoning/--no-rag-reasoning", 
                                           help="Use RAG references for invariant and ranking function inference; Default is enabled")
) -> None:
    """Analyze a source snippet for termination likelihood."""

    # Check file extension
    suffix = code_file.suffix.lower()
    is_cpp = suffix in {".c", ".cpp", ".h", ".hpp", ".cc", ".cxx"}
    
    if not is_cpp and not enable_translation:
        console.print(f"[bold red]Error:[/bold red] File '{code_file.name}' does not appear to be a C/C++ file.")
        console.print("Please use [bold]--enable-translation[/bold] to enable automatic translation.")
        raise typer.Exit(code=1)

    pipeline = TerminationPipeline(enable_translation=enable_translation, 
                                   knowledge_base_path=str(knowledge_base) if knowledge_base else None
                                   )
    code = code_file.read_text(encoding="utf-8")
    result = pipeline.analyze(code, top_k=top_k, use_rag_in_reasoning=use_rag_reasoning)

    # Show translation info if applicable
    if enable_translation and result.report_path:
        import json
        with open(result.report_path, 'r', encoding='utf-8') as f:
            report = json.load(f)
        translation = report.get("translation", {})
        if translation.get("translated"):
            console.print("[bold cyan]✓ Code was translated to C++[/bold cyan]")

    console.rule("Prediction")
    console.print(f"Label: [bold]{result.label}[/bold]")
    console.print(f"Reasoning: {result.reasoning}")
    
    if result.verification_result:
        console.print(f"Verification: [bold]{result.verification_result}[/bold]")
    if result.ranking_function:
        console.print(f"Ranking Function: [italic]{result.ranking_function}[/italic]")
    if result.invariants:
        console.print("Invariants:")
        for inv in result.invariants:
            console.print(f"  - {inv}")

    console.print(f"Report saved at: {result.report_path}")

    table = Table(title="Referenced cases")
    table.add_column("Case ID")
    table.add_column("Label")
    table.add_column("Similarity")
    for ref in result.references:
        similarity = ref.metadata.get("similarity", "n/a")
        if isinstance(similarity, float):
            similarity_str = f"{similarity:.3f}"
        else:
            similarity_str = str(similarity)
        table.add_row(ref.case_id, ref.label, similarity_str)
    console.print(table)


@app.command()
def batch_analyze(
    input_dir: Path = typer.Option(..., exists=True, file_okay=False, dir_okay=True, help="Directory containing source files"),
    top_k: int = 5,
    enable_translation: bool = typer.Option(False, "--enable-translation", "-t", help="Enable LLM-based translation"),
    knowledge_base: Optional[Path] = typer.Option(None, "--kb", help="Path to a custom knowledge base JSON file"),
    recursive: bool = typer.Option(False, "--recursive", "-r", help="Recursively search for files"),
    use_rag_reasoning: bool = typer.Option(True, "--use-rag-reasoning/--no-rag-reasoning", help="Use RAG references for invariant and ranking function inference")
) -> None:
    """Batch analyze all C/C++ files in a directory."""
    
    import csv
    from datetime import datetime
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
    
    # Find all C/C++ files
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
    
    pipeline = TerminationPipeline(
        enable_translation=enable_translation,
        knowledge_base_path=str(knowledge_base) if knowledge_base else None
    )
    
    # Prepare CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = input_dir / f"batch_report_{timestamp}.csv"
    csv_headers = [
        "Filename", "Relative Path", "Run ID", "Date Time", "Duration (s)", 
        "LLM Calls", "Label", "RAG Similarity", "Invariants", 
        "Ranking Function", "Z3 Result", "Report Path", "Error"
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
                    result = pipeline.analyze(code, top_k=top_k, use_rag_in_reasoning=use_rag_reasoning)
                    results.append((file_path.name, result))
                    
                    # Write to CSV
                    rag_sim = result.references[0].metadata.get("similarity", 0.0) if result.references else 0.0
                    invariants_str = "; ".join(result.invariants) if result.invariants else ""
                    
                    row = [
                        file_path.name,
                        rel_path,
                        result.run_id,
                        datetime.now().isoformat(),
                        f"{result.duration_seconds:.2f}",
                        result.llm_calls,
                        result.label,
                        rag_sim,
                        invariants_str,
                        result.ranking_function or "",
                        result.verification_result or "",
                        str(result.report_path) if result.report_path else "",
                        ""
                    ]
                    writer.writerow(row)
                    csvfile.flush()
                    
                except Exception as e:
                    console.print(f"[red]Error analyzing {file_path.name}: {e}[/red]")
                    # Still record the failure in CSV so batch runs are auditable.
                    row = [
                        file_path.name,
                        rel_path,
                        "",
                        datetime.now().isoformat(),
                        "",
                        "",
                        "error",
                        "",
                        "",
                        "",
                        "",
                        "",
                        str(e),
                    ]
                    writer.writerow(row)
                    csvfile.flush()
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


@app.command()
def review(
    code_file: Path = typer.Option(..., exists=True, readable=True),
    label: str = typer.Option(..., help="terminating|non-terminating|unknown"),
    explanation: str = typer.Option(..., help="Brief reasoning"),
    reviewer: Optional[str] = typer.Option(None, help="Reviewer identifier"),
) -> None:
    """Add a manually reviewed case back into the RAG store."""

    pipeline = TerminationPipeline()
    code = code_file.read_text(encoding="utf-8")
    label_value = label.lower()
    if label_value not in VALID_LABELS:
        raise typer.BadParameter(f"label must be one of {', '.join(sorted(VALID_LABELS))}")
    pending = PendingReviewCase(
        code=code,
        label=label_value,  # type: ignore[assignment]
        explanation=explanation,
        loops=pipeline.loop_extractor.extract(code),
        reviewer=reviewer or "cli",
    )
    case = pipeline.ingest_reviewed_case(pending)
    console.print(f"Stored case {case.case_id}. Pending rebuild: {pipeline.knowledge_base.needs_rebuild()}")


@app.command()
def rebuild_index() -> None:
    """Force a full HNSW rebuild from the current knowledge base."""

    pipeline = TerminationPipeline()
    pipeline.index_manager.rebuild(pipeline.knowledge_base.cases)
    pipeline.knowledge_base.mark_rebuilt()
    console.print("Index rebuilt and persisted.")
