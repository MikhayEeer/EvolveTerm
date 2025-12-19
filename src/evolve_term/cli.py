"""Typer-based CLI to interact with the termination pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, List, Dict, Any
import json
import sys

import typer
from rich.console import Console
from rich.table import Table

from .models import PendingReviewCase, KnowledgeCase
from .pipeline import TerminationPipeline
from .translator import CodeTranslator
from .loop_extractor import LoopExtractor
from .predict import Predictor
from .verifier import Z3Verifier
from .prompts_loader import PromptRepository
from .llm_client import build_llm_client

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
                                           help="Use RAG references for invariant and ranking function inference; Default is enabled"),
    svm_ranker_path: Optional[Path] = typer.Option(None, "--svm-ranker", help="Path to SVMRanker root directory"),
    known_terminating: bool = typer.Option(False, "--known-terminating", help="Hint that the program is known to terminate")
) -> None:
    """Analyze a source snippet for termination likelihood."""

    # Check file extension
    suffix = code_file.suffix.lower()
    is_cpp = suffix in {".c", ".cpp", ".h", ".hpp", ".cc", ".cxx"}
    
    if not is_cpp and not enable_translation:
        console.print(f"[bold red]Error:[/bold red] File '{code_file.name}' does not appear to be a C/C++ file.")
        console.print("Please use [bold]--enable-translation[/bold] to enable automatic translation.")
        raise typer.Exit(code=1)

    pipeline = TerminationPipeline(
        enable_translation=enable_translation, 
        knowledge_base_path=str(knowledge_base) if knowledge_base else None,
        svm_ranker_path=str(svm_ranker_path) if svm_ranker_path else None
    )
    code = code_file.read_text(encoding="utf-8")
    result = pipeline.analyze(
        code, 
        top_k=top_k, 
        use_rag_in_reasoning=use_rag_reasoning,
        use_svm_ranker=bool(svm_ranker_path),
        known_terminating=known_terminating
    )

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
    use_rag_reasoning: bool = typer.Option(True, "--use-rag-reasoning/--no-rag-reasoning", help="Use RAG references for invariant and ranking function inference"),
    svm_ranker_path: Optional[Path] = typer.Option(None, "--svm-ranker", help="Path to SVMRanker root directory"),
    known_terminating: bool = typer.Option(False, "--known-terminating", help="Hint that the program is known to terminate")
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
        knowledge_base_path=str(knowledge_base) if knowledge_base else None,
        svm_ranker_path=str(svm_ranker_path) if svm_ranker_path else None
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
                    result = pipeline.analyze(
                        code, 
                        top_k=top_k, 
                        use_rag_in_reasoning=use_rag_reasoning,
                        use_svm_ranker=bool(svm_ranker_path),
                        known_terminating=known_terminating
                    )
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

@app.command()
def translate(
    input: Path = typer.Option(..., exists=True, help="Input file or directory"),
    output: Optional[Path] = typer.Option(None, help="Output file or directory"),
    llm_config: str = typer.Option("llm_config.json", help="Path to LLM config"),
    recursive: bool = typer.Option(False, "--recursive", "-r", help="Recursively search for files if input is directory"),
) -> None:
    """
    Translate code to C++.
    
    Input:
        - Single file: Source code file (e.g., .java, .py).
        - Directory: Directory containing source files.
        
    Output:
        - Single file: Translated C++ code.
        - Directory: Translated files with same structure or flat.
    """
    translator = CodeTranslator(config_name=llm_config)
    
    if input.is_file():
        console.print(f"Translating {input}...")
        code = input.read_text(encoding="utf-8")
        translated = translator.translate(code)
        
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
        files = list(input.rglob("*") if recursive else input.glob("*"))
        files = [f for f in files if f.is_file()]
        console.print(f"Found {len(files)} files to translate.")
        
        if output and not output.exists():
            output.mkdir(parents=True)
            
        for f in files:
            try:
                console.print(f"Translating {f.name}...")
                code = f.read_text(encoding="utf-8")
                translated = translator.translate(code)
                
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
                console.print(f"[red]Error translating {f.name}: {e}[/red]")


@app.command()
def extract(
    input: Path = typer.Option(..., exists=True, help="Input file or directory"),
    output: Optional[Path] = typer.Option(None, help="Output file or directory"),
    llm_config: str = typer.Option("llm_config.json", help="Path to LLM config"),
    recursive: bool = typer.Option(False, "--recursive", "-r", help="Recursively search for files if input is directory"),
) -> None:
    """
    Extract loops from code.
    
    Output Format (JSON):
    [
        "loop_code_1",
        "loop_code_2"
    ]
    """
    llm_client = build_llm_client(llm_config)
    prompt_repo = PromptRepository()
    extractor = LoopExtractor(llm_client, prompt_repo)
    
    def process_file(f: Path) -> List[str]:
        code = f.read_text(encoding="utf-8")
        return extractor.extract(code)

    if input.is_file():
        console.print(f"Extracting loops from {input}...")
        loops = process_file(input)
        
        if output:
            if output.is_dir():
                out_path = output / (input.stem + ".json")
            else:
                out_path = output
            out_path.write_text(json.dumps(loops, indent=2), encoding="utf-8")
            console.print(f"Saved to {out_path}")
        else:
            console.print(json.dumps(loops, indent=2))
            
    elif input.is_dir():
        files = list(input.rglob("*") if recursive else input.glob("*"))
        files = [f for f in files if f.is_file()]
        console.print(f"Found {len(files)} files.")
        
        if output and not output.exists():
            output.mkdir(parents=True)
            
        for f in files:
            try:
                console.print(f"Extracting {f.name}...")
                loops = process_file(f)
                
                if output:
                    try:
                        rel_path = f.relative_to(input)
                        out_path = output / rel_path.with_suffix(".json")
                    except ValueError:
                        out_path = output / (f.stem + ".json")
                    
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    out_path.write_text(json.dumps(loops, indent=2), encoding="utf-8")
                else:
                    console.print(f"--- {f.name} ---")
                    console.print(json.dumps(loops, indent=2))
            except Exception as e:
                console.print(f"[red]Error extracting {f.name}: {e}[/red]")


@app.command()
def invariant(
    input: Path = typer.Option(..., exists=True, help="Input code file or directory"),
    references_file: Optional[Path] = typer.Option(None, help="JSON file containing reference cases (List[KnowledgeCase dict])"),
    output: Optional[Path] = typer.Option(None, help="Output file or directory"),
    llm_config: str = typer.Option("llm_config.json", help="Path to LLM config"),
    recursive: bool = typer.Option(False, "--recursive", "-r", help="Recursively search for files if input is directory"),
) -> None:
    """
    Infer invariants for the given code.
    
    References JSON Format:
    [
        {
            "case_id": "...",
            "code": "...",
            "label": "...",
            "explanation": "...",
            "loops": ["..."],
            "metadata": {...}
        },
        ...
    ]
    """
    llm_client = build_llm_client(llm_config)
    prompt_repo = PromptRepository()
    predictor = Predictor(llm_client, prompt_repo)
    
    references = []
    if references_file:
        data = json.loads(references_file.read_text(encoding="utf-8"))
        references = [KnowledgeCase(**item) for item in data]

    def process_file(f: Path) -> List[str]:
        code = f.read_text(encoding="utf-8")
        # Note: In standalone mode, we might not have specific references per file unless provided.
        # Here we use the global references provided via CLI.
        return predictor.infer_invariants(code, references)

    if input.is_file():
        console.print(f"Inferring invariants for {input}...")
        invariants = process_file(input)
        
        if output:
            if output.is_dir():
                out_path = output / (input.stem + ".json")
            else:
                out_path = output
            out_path.write_text(json.dumps(invariants, indent=2), encoding="utf-8")
            console.print(f"Saved to {out_path}")
        else:
            console.print(json.dumps(invariants, indent=2))
            
    elif input.is_dir():
        files = list(input.rglob("*") if recursive else input.glob("*"))
        files = [f for f in files if f.is_file()]
        
        if output and not output.exists():
            output.mkdir(parents=True)
            
        for f in files:
            try:
                console.print(f"Processing {f.name}...")
                invariants = process_file(f)
                
                if output:
                    try:
                        rel_path = f.relative_to(input)
                        out_path = output / rel_path.with_suffix(".json")
                    except ValueError:
                        out_path = output / (f.stem + ".json")
                    
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    out_path.write_text(json.dumps(invariants, indent=2), encoding="utf-8")
                else:
                    console.print(f"--- {f.name} ---")
                    console.print(json.dumps(invariants, indent=2))
            except Exception as e:
                console.print(f"[red]Error processing {f.name}: {e}[/red]")


@app.command()
def ranking(
    input: Path = typer.Option(..., exists=True, help="Input code file or directory"),
    invariants_file: Optional[Path] = typer.Option(None, help="JSON file containing invariants list (for single file input)"),
    references_file: Optional[Path] = typer.Option(None, help="JSON file containing reference cases"),
    output: Optional[Path] = typer.Option(None, help="Output file or directory"),
    llm_config: str = typer.Option("llm_config.json", help="Path to LLM config"),
    recursive: bool = typer.Option(False, "--recursive", "-r", help="Recursively search for files if input is directory"),
) -> None:
    """
    Infer ranking function for the given code.
    
    Invariants JSON Format:
    ["inv1", "inv2", ...]
    """
    llm_client = build_llm_client(llm_config)
    prompt_repo = PromptRepository()
    predictor = Predictor(llm_client, prompt_repo)
    
    references = []
    if references_file:
        data = json.loads(references_file.read_text(encoding="utf-8"))
        references = [KnowledgeCase(**item) for item in data]

    def process_file(f: Path, invs: List[str]) -> Dict[str, str]:
        code = f.read_text(encoding="utf-8")
        rf, explanation = predictor.infer_ranking(code, invs, references)
        return {"ranking_function": rf, "explanation": explanation}

    if input.is_file():
        console.print(f"Inferring ranking function for {input}...")
        invariants = []
        if invariants_file:
            invariants = json.loads(invariants_file.read_text(encoding="utf-8"))
            
        result = process_file(input, invariants)
        
        if output:
            if output.is_dir():
                out_path = output / (input.stem + ".json")
            else:
                out_path = output
            out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
            console.print(f"Saved to {out_path}")
        else:
            console.print(json.dumps(result, indent=2))
            
    elif input.is_dir():
        # In batch mode, we assume no invariants are provided externally for each file, 
        # or we could support a mapping file, but for now let's assume empty invariants 
        # or that the user just wants to test RF generation without invariants.
        if invariants_file:
            console.print("[yellow]Warning: --invariants-file ignored in batch mode (cannot map single invariant list to multiple files).[/yellow]")
        
        files = list(input.rglob("*") if recursive else input.glob("*"))
        files = [f for f in files if f.is_file()]
        
        if output and not output.exists():
            output.mkdir(parents=True)
            
        for f in files:
            try:
                console.print(f"Processing {f.name}...")
                result = process_file(f, [])
                
                if output:
                    try:
                        rel_path = f.relative_to(input)
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


@app.command()
def verify(
    input: Path = typer.Option(..., exists=True, help="Input code file or directory"),
    ranking_func: Optional[str] = typer.Option(None, help="Ranking function string (for single file)"),
    ranking_file: Optional[Path] = typer.Option(None, help="File containing ranking function (JSON {ranking_function: ...} or raw text)"),
    invariants_file: Optional[Path] = typer.Option(None, help="JSON file containing invariants list"),
    output: Optional[Path] = typer.Option(None, help="Output file or directory"),
    llm_config: str = typer.Option("llm_config.json", help="Path to LLM config"),
    recursive: bool = typer.Option(False, "--recursive", "-r", help="Recursively search for files if input is directory"),
) -> None:
    """
    Verify ranking function and invariants using Z3.
    """
    llm_client = build_llm_client(llm_config)
    prompt_repo = PromptRepository()
    verifier = Z3Verifier(llm_client, prompt_repo)
    
    def get_rf(f_path: Path) -> str | None:
        # If explicit string provided, use it (only valid for single file really)
        if ranking_func:
            return ranking_func
        
        # If ranking file provided
        if ranking_file:
            # If ranking_file is a directory, try to find corresponding file
            if ranking_file.is_dir():
                # Try to find file with same stem
                # This is a simple heuristic for batch mode
                candidate = ranking_file / (f_path.stem + ".json")
                if not candidate.exists():
                    candidate = ranking_file / (f_path.stem + ".txt")
                
                if candidate.exists():
                    content = candidate.read_text(encoding="utf-8")
                    try:
                        data = json.loads(content)
                        if isinstance(data, dict):
                            return data.get("ranking_function")
                        return str(data)
                    except:
                        return content.strip()
                return None
            else:
                # Single ranking file provided
                content = ranking_file.read_text(encoding="utf-8")
                try:
                    data = json.loads(content)
                    if isinstance(data, dict):
                        return data.get("ranking_function")
                    return str(data)
                except:
                    return content.strip()
        return None

    def get_invs(f_path: Path) -> List[str]:
        if invariants_file:
            if invariants_file.is_dir():
                candidate = invariants_file / (f_path.stem + ".json")
                if candidate.exists():
                    try:
                        return json.loads(candidate.read_text(encoding="utf-8"))
                    except:
                        return []
            else:
                try:
                    return json.loads(invariants_file.read_text(encoding="utf-8"))
                except:
                    return []
        return []

    def process_file(f: Path) -> str:
        code = f.read_text(encoding="utf-8")
        rf = get_rf(f)
        invs = get_invs(f)
        
        if not rf:
            return "Skipped (No Ranking Function)"
            
        return verifier.verify(code, invs, rf)

    if input.is_file():
        console.print(f"Verifying {input}...")
        result = process_file(input)
        
        if output:
            if output.is_dir():
                out_path = output / (input.stem + ".txt")
            else:
                out_path = output
            out_path.write_text(result, encoding="utf-8")
            console.print(f"Saved to {out_path}")
        else:
            console.print(f"Result: {result}")
            
    elif input.is_dir():
        files = list(input.rglob("*") if recursive else input.glob("*"))
        files = [f for f in files if f.is_file()]
        
        if output and not output.exists():
            output.mkdir(parents=True)
            
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
                console.print(f"[red]Error verifying {f.name}: {e}[/red]")
