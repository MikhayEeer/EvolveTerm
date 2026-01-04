"""Typer-based CLI to interact with the termination pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, List, Dict, Any
import json
import sys
import re
import yaml
import tempfile
from datetime import datetime

import typer
from rich.console import Console
from rich.table import Table

from .models import PendingReviewCase
from .pipeline import TerminationPipeline
from .translator import CodeTranslator
from .loop_extractor import LoopExtractor
from .predict import Predictor
from .verifier import Z3Verifier
from .svm_ranker import SVMRankerClient
from .cli_utils import ensure_output_dir, collect_files, load_references
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
    known_terminating: bool = typer.Option(False, "--known-terminating", help="Hint that the program is known to terminate"),
    # Ablation parameters
    extraction_prompt_version: str = typer.Option("v2", "--prompt-version", "-p", help="Prompt version for loop extraction (v1 or v2)"),
    use_loops_for_embedding: bool = typer.Option(True, "--embed-loops/--embed-code", help="Use extracted loops for embedding vs full code"),
    use_loops_for_reasoning: bool = typer.Option(True, "--reason-loops/--reason-code", help="Use extracted loops for reasoning vs full code")
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
        known_terminating=known_terminating,
        extraction_prompt_version=extraction_prompt_version,
        use_loops_for_embedding=use_loops_for_embedding,
        use_loops_for_reasoning=use_loops_for_reasoning
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
    known_terminating: bool = typer.Option(False, "--known-terminating", help="Hint that the program is known to terminate"),
    # Ablation parameters
    extraction_prompt_version: str = typer.Option("v2", "--prompt-version", "-p", help="Prompt version for loop extraction (v1 or v2)"),
    use_loops_for_embedding: bool = typer.Option(True, "--embed-loops/--embed-code", help="Use extracted loops for embedding vs full code"),
    use_loops_for_reasoning: bool = typer.Option(True, "--reason-loops/--reason-code", help="Use extracted loops for reasoning vs full code")
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
                        known_terminating=known_terminating,
                        extraction_prompt_version=extraction_prompt_version,
                        use_loops_for_embedding=use_loops_for_embedding,
                        use_loops_for_reasoning=use_loops_for_reasoning
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
        files = collect_files(input, recursive)
        console.print(f"Found {len(files)} files to translate.")
        
        ensure_output_dir(output)
            
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
    output: Optional[Path] = typer.Option(None, help="Output file or directory (Deprecated for YAML output)"),
    llm_config: str = typer.Option("llm_config.json", help="Path to LLM config"),
    recursive: bool = typer.Option(False, "--recursive", "-r", help="Recursively search for files if input is directory"),
    prompt_version: str = typer.Option("v1", "--prompt-version", "-p", help="Prompt version (v1 or v2)"),
) -> None:
    """
    Extract loops from code and save to YAML.
    
    Output:
        Saves a YAML file for each input file in a 'extract_result' subdirectory 
        next to the input file.
        Filename format: {filename}_pmt_yaml{version}_{model}_auto.yml
    """
    # Load config to get model details
    config_path = Path(llm_config)
    model_name = "unknown"
    model_config = {}
    if config_path.exists():
        try:
            model_config = json.loads(config_path.read_text(encoding="utf-8"))
            model_name = model_config.get("model_name", model_config.get("model", "unknown"))
        except:
            pass

    llm_client = build_llm_client(llm_config)
    # Try to get more accurate model name from client if available
    if hasattr(llm_client, "model"):
        model_name = llm_client.model
    elif hasattr(llm_client, "model_name"):
        model_name = llm_client.model_name

    prompt_repo = PromptRepository()
    extractor = LoopExtractor(llm_client, prompt_repo)
    
    # Determine prompt name based on version
    prompt_name = f"loop_extraction/yaml_{prompt_version}"
    prompt_type = f"yaml{prompt_version}"

    # Custom Dumper for block style strings
    class LiteralDumper(yaml.SafeDumper):
        def represent_scalar(self, tag, value, style=None):
            if "\n" in value and tag == 'tag:yaml.org,2002:str':
                style = '|'
            return super().represent_scalar(tag, value, style)

    def process_file(f: Path, base_dir: Optional[Path], output_root: Optional[Path]):
        code = f.read_text(encoding="utf-8")
        loops = extractor.extract(code, prompt_name=prompt_name)
        
        timestamp = datetime.now().strftime("%Y-%m-%dT%H:%M")
        safe_model_name = re.sub(r'[^\w\-]', '', str(model_name))
        
        yaml_data = {
            "source_file": f.name,
            "source_path": str(f.relative_to(base_dir)) if base_dir else str(f),
            "basic": {
                "name": safe_model_name,
                "type": model_config.get("type", "llm"),
                "prompt": f"prompts/{prompt_name}",
                "config": model_config,
                "call_type": "cli_extract",
                "time": timestamp
            },
            "loops": [
                {
                    "id": i + 1,
                    "code": loop.replace('\t', '    ')
                }
                for i, loop in enumerate(loops)
            ]
        }
        
        # Output directory: sibling 'extract_result'
        result_dir = f.parent / "extract_result"
        result_dir.mkdir(exist_ok=True)
        
        filename = f"{f.stem}_pmt_{prompt_type}_{safe_model_name}_auto.yml"
        out_path = result_dir / filename
        
        with open(out_path, 'w', encoding='utf-8') as yf:
            yaml.dump(yaml_data, yf, Dumper=LiteralDumper, sort_keys=False, allow_unicode=True)
            
        console.print(f"Saved extraction to {out_path}")

    if input.is_file():
        console.print(f"Extracting loops from {input}...")
        process_file(input, None)
            
    elif input.is_dir():
        extensions = {".c", ".cpp", ".h", ".hpp", ".cc", ".cxx"}
        files = collect_files(input, recursive, extensions=extensions)
        console.print(f"Found {len(files)} files.")
        
        for f in files:
            try:
                process_file(f, input, None)
            except Exception as e:
                console.print(f"[red]Error extracting {f.name}: {e}[/red]")


@app.command()
def invariant(
    input: Path = typer.Option(..., exists=True, help="Input code file, YAML file, or directory"),
    references_file: Optional[Path] = typer.Option(None, help="JSON file containing reference cases (List[KnowledgeCase dict])"),
    output: Optional[Path] = typer.Option(None, help="Output file or directory"),
    llm_config: str = typer.Option("llm_config.json", help="Path to LLM config"),
    recursive: bool = typer.Option(False, "--recursive", "-r", help="Recursively search for files if input is directory"),
    mode: str = typer.Option("auto", "--mode", "-m", help="Filter mode for batch processing: 'auto' (all), 'yaml' (only extract results), 'code' (only C/C++ files)"),
) -> None:
    """
    Infer invariants for the given code or extracted YAML.
    
    Input can be:
    1. A C/C++ source file.
    2. A YAML file generated by the 'extract' command.
    3. A directory containing such files.

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
    # Load config to get model details
    config_path = Path(llm_config)
    model_name = "unknown"
    model_config = {}
    if config_path.exists():
        try:
            model_config = json.loads(config_path.read_text(encoding="utf-8"))
            model_name = model_config.get("model_name", model_config.get("model", "unknown"))
        except:
            pass

    llm_client = build_llm_client(llm_config)
    # Try to get more accurate model name from client if available
    if hasattr(llm_client, "model"):
        model_name = llm_client.model
    elif hasattr(llm_client, "model_name"):
        model_name = llm_client.model_name

    prompt_repo = PromptRepository()
    predictor = Predictor(llm_client, prompt_repo)
    
    references = load_references(references_file)

    # Custom Dumper for block style strings
    class LiteralDumper(yaml.SafeDumper):
        def represent_scalar(self, tag, value, style=None):
            if "\n" in value and tag == 'tag:yaml.org,2002:str':
                style = '|'
            return super().represent_scalar(tag, value, style)

    def process_file(f: Path, base_dir: Optional[Path], output_root: Optional[Path]):
        loops_to_analyze = []
        source_type = "code"
        source_path = str(f.relative_to(base_dir)) if base_dir else str(f)
        
        # Determine input type
        if f.suffix.lower() in {'.yml', '.yaml'}:
            source_type = "yaml"
            try:
                data = yaml.safe_load(f.read_text(encoding="utf-8"))
                if isinstance(data, dict) and "source_path" in data:
                    source_path = data["source_path"]
                if "loops" in data:
                    loops_to_analyze = [item["code"] for item in data["loops"]]
                else:
                    # Not an extract result, skip
                    return
            except Exception as e:
                console.print(f"[red]Error parsing YAML {f}: {e}[/red]")
                return
        else:
            # Treat as raw code file
            code = f.read_text(encoding="utf-8")
            # For raw code, we treat the whole file as one context unless we extract loops first.
            # But 'invariant' command is low-level. Let's assume user wants to analyze the file content.
            loops_to_analyze = [code]

        all_invariants = []
        for i, loop_code in enumerate(loops_to_analyze):
            invariants = predictor.infer_invariants(loop_code, references)
            all_invariants.append({
                "loop_id": i + 1,
                "code": loop_code,
                "invariants": invariants
            })
            console.print(f"[green]Inferred {len(invariants)} invariants for loop {i+1} in {f.name}[/green]")

        # Output generation
        timestamp = datetime.now().strftime("%Y-%m-%dT%H:%M")
        safe_model_name = re.sub(r'[^\w\-]', '', str(model_name))
        
        yaml_data = {
            "source_file": f.name,
            "source_path": source_path,
            "basic": {
                "name": safe_model_name,
                "type": model_config.get("type", "llm"),
                "task": "invariant_inference",
                "config": model_config,
                "time": timestamp
            },
            "invariants_result": all_invariants
        }
        
        filename = f"{f.stem}_inv_{safe_model_name}_auto.yml"
        if output_root:
            rel_parent = Path(".") if base_dir is None else f.parent.relative_to(base_dir)
            result_dir = output_root / rel_parent / "invariant_result"
            result_dir.mkdir(parents=True, exist_ok=True)
            out_path = result_dir / filename
        else:
            # Output directory: sibling 'invariant_result'
            result_dir = f.parent / "invariant_result"
            result_dir.mkdir(exist_ok=True)
            out_path = result_dir / filename
        
        with open(out_path, 'w', encoding='utf-8') as yf:
            yaml.dump(yaml_data, yf, Dumper=LiteralDumper, sort_keys=False, allow_unicode=True)
            
        console.print(f"Saved invariants to {out_path}")

    if input.is_file():
        output_root = None
        if output and output.suffix not in {".yml", ".yaml"}:
            if not output.exists():
                output.mkdir(parents=True)
            output_root = output
        process_file(input, None, output_root)
    elif input.is_dir():
        # Find both code and yaml files
        code_extensions = {".c", ".cpp", ".h", ".hpp", ".cc", ".cxx"}
        yaml_extensions = {".yml", ".yaml"}
        
        files = collect_files(input, recursive)
        
        filtered_files = []
        for f in files:
            ext = f.suffix.lower()
            if mode == "code":
                if ext in code_extensions:
                    filtered_files.append(f)
            elif mode == "yaml":
                if ext in yaml_extensions:
                    # Optional: Check if it's an extract result by filename pattern or content?
                    # Filename check is faster: usually contains "_pmt_" or similar
                    # But content check is safer. Let's rely on process_file to skip invalid YAMLs
                    filtered_files.append(f)
            else: # auto
                if ext in code_extensions or ext in yaml_extensions:
                    filtered_files.append(f)
        
        console.print(f"Found {len(filtered_files)} files to analyze (mode={mode}).")
        if output and not output.exists():
            output.mkdir(parents=True)
        for f in filtered_files:
            try:
                process_file(f, input, output)
            except Exception as e:
                console.print(f"[red]Error processing {f.name}: {e}[/red]")
        return

    return

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
        files = collect_files(input, recursive)
        
        ensure_output_dir(output)
            
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
    input: Path = typer.Option(..., exists=True, help="Input code file, YAML file, or directory"),
    invariants_file: Optional[Path] = typer.Option(None, help="JSON file containing invariants list (for single file input)"),
    references_file: Optional[Path] = typer.Option(None, help="JSON file containing reference cases"),
    output: Optional[Path] = typer.Option(None, help="Output file or directory"),
    llm_config: str = typer.Option("llm_config.json", help="Path to LLM config"),
    recursive: bool = typer.Option(False, "--recursive", "-r", help="Recursively search for files if input is directory"),
    mode: str = typer.Option("auto", "--mode", "-m", help="Filter mode for batch processing: 'auto' (all), 'yaml' (only extract/invariant results), 'code' (only C/C++ files)"),
    ranking_mode: str = typer.Option(
        "direct",
        "--ranking-mode",
        help="Ranking prompt mode: 'direct', 'template', or 'template-known'",
    ),
) -> None:
    """
    Infer ranking function for the given code.
    
    Input can be:
    1. A C/C++ source file.
    2. A YAML file generated by 'extract' or 'invariant' command.
    3. A directory containing such files.
    
    Invariants JSON Format (for single file):
    ["inv1", "inv2", ...]
    """
    llm_client = build_llm_client(llm_config)
    prompt_repo = PromptRepository()
    predictor = Predictor(llm_client, prompt_repo)

    ranking_mode_value = ranking_mode.strip().lower()
    if ranking_mode_value in {"template-known", "template_known"}:
        rf_mode = "template"
        rf_known_terminating = True
    elif ranking_mode_value in {"direct", "template"}:
        rf_mode = ranking_mode_value
        rf_known_terminating = False
    else:
        raise typer.BadParameter("ranking_mode must be one of: direct, template, template-known")
    
    references = load_references(references_file)

    def process_file(f: Path, invs: List[str]) -> Dict[str, Any]:
        code = f.read_text(encoding="utf-8")
        rf, explanation, metadata = predictor.infer_ranking(
            code, invs, references, mode=rf_mode, known_terminating=rf_known_terminating
        )
        if rf_mode == "template":
            return {
                "template_type": metadata.get("type"),
                "template_depth": metadata.get("depth"),
                "explanation": explanation,
            }
        return {"ranking_function": rf, "explanation": explanation}

    def process_yaml_input(f: Path) -> tuple[List[Dict[str, Any]], str]:
        try:
            content = yaml.safe_load(f.read_text(encoding="utf-8"))
        except Exception as e:
            console.print(f"[red]Error parsing YAML {f}: {e}[/red]")
            return [], str(f)

        results = []
        source_path = content.get("source_path") if isinstance(content, dict) else None
        # Case 1: Invariant Result YAML
        if "invariants_result" in content:
            console.print(f"[blue]Detected Invariant Result YAML: {f.name}[/blue]")
            for item in content["invariants_result"]:
                loop_id = item.get("loop_id") or item.get("id")
                code = item.get("code", "")
                invs = item.get("invariants", [])
                
                console.print(f"  Inferring ranking for Loop {loop_id}...")
                rf, explanation, metadata = predictor.infer_ranking(
                    code, invs, references, mode=rf_mode, known_terminating=rf_known_terminating
                )
                result_entry = {
                    "loop_id": loop_id,
                    "code": code,
                    "invariants": invs,
                    "explanation": explanation,
                }
                if rf_mode == "template":
                    result_entry.update(
                        {
                            "template_type": metadata.get("type"),
                            "template_depth": metadata.get("depth"),
                        }
                    )
                else:
                    result_entry["ranking_function"] = rf
                results.append(result_entry)
                
        # Case 2: Extract Result YAML
        elif "loops" in content:
            console.print(f"[blue]Detected Extract Result YAML: {f.name}[/blue]")
            for item in content["loops"]:
                loop_id = item.get("id")
                code = item.get("code", "")
                # No invariants in extract result
                
                console.print(f"  Inferring ranking for Loop {loop_id}...")
                rf, explanation, metadata = predictor.infer_ranking(
                    code, [], references, mode=rf_mode, known_terminating=rf_known_terminating
                )
                result_entry = {
                    "loop_id": loop_id,
                    "code": code,
                    "invariants": [],
                    "explanation": explanation,
                }
                if rf_mode == "template":
                    result_entry.update(
                        {
                            "template_type": metadata.get("type"),
                            "template_depth": metadata.get("depth"),
                        }
                    )
                else:
                    result_entry["ranking_function"] = rf
                results.append(result_entry)
        else:
            console.print(f"[yellow]Unknown YAML format in {f.name}. Expected 'loops' or 'invariants_result'.[/yellow]")
            
        return results, source_path or str(f)

    if input.is_file():
        if input.suffix.lower() in {'.yml', '.yaml'}:
            results, source_path = process_yaml_input(input)
            if output:
                if output.is_dir():
                    out_path = output / (input.stem + "_ranking.yml")
                else:
                    out_path = output
                
                # Wrap in a structure
                out_data = {
                    "source_file": input.name,
                    "source_path": source_path,
                    "task": "ranking_inference",
                    "ranking_results": results
                }
                with open(out_path, 'w', encoding='utf-8') as f:
                    yaml.dump(out_data, f, sort_keys=False, allow_unicode=True)
                console.print(f"Saved ranking results to {out_path}")
            else:
                console.print(yaml.dump(results, sort_keys=False, allow_unicode=True))
        else:
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
        
        code_extensions = {".c", ".cpp", ".h", ".hpp", ".cc", ".cxx"}
        yaml_extensions = {".yml", ".yaml"}
        
        files = collect_files(input, recursive)
        
        filtered_files = []
        for f in files:
            ext = f.suffix.lower()
            if mode == "code":
                if ext in code_extensions:
                    filtered_files.append(f)
            elif mode == "yaml":
                if ext in yaml_extensions:
                    filtered_files.append(f)
            else: # auto
                if ext in code_extensions or ext in yaml_extensions:
                    filtered_files.append(f)
        
        console.print(f"Found {len(filtered_files)} files to analyze (mode={mode}).")
        
        ensure_output_dir(output)
            
        for f in filtered_files:
            try:
                console.print(f"Processing {f.name}...")
                
                if f.suffix.lower() in yaml_extensions:
                    results, source_path = process_yaml_input(f)
                    if output:
                        try:
                            rel_path = f.relative_to(input)
                            out_path = output / rel_path.with_suffix(".yml")
                            # Append _ranking suffix to distinguish
                            out_path = out_path.parent / (out_path.stem + "_ranking.yml")
                        except ValueError:
                            out_path = output / (f.stem + "_ranking.yml")
                        
                        out_path.parent.mkdir(parents=True, exist_ok=True)
                        out_data = {
                            "source_file": f.name,
                            "source_path": source_path,
                            "task": "ranking_inference",
                            "ranking_results": results
                        }
                        with open(out_path, 'w', encoding='utf-8') as yf:
                            yaml.dump(out_data, yf, sort_keys=False, allow_unicode=True)
                    else:
                        console.print(f"--- {f.name} ---")
                        console.print(yaml.dump(results, sort_keys=False, allow_unicode=True))
                else:
                    # C/C++ file
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
def predict(
    input: Path = typer.Option(..., exists=True, help="Input code file, directory, or YAML"),
    references_file: Optional[Path] = typer.Option(None, help="JSON file containing reference cases"),
    llm_config: str = typer.Option("llm_config.json", help="Path to LLM config"),
    output: Optional[Path] = typer.Option(None, help="Output file or directory"),
    recursive: bool = typer.Option(False, "--recursive", "-r", help="Recursively search for files if input is directory"),
) -> None:
    """
    Perform preliminary termination prediction.
    Supports C files (auto-extract loops) or YAML files (from extract/invariant modules).
    """
    llm_client = build_llm_client(llm_config)
    prompt_repo = PromptRepository()
    predictor = Predictor(llm_client, prompt_repo)
    loop_extractor = LoopExtractor(llm_client, prompt_repo)
    
    references = load_references(references_file)

    def process_yaml_input(f: Path) -> List[Dict[str, Any]]:
        try:
            content = yaml.safe_load(f.read_text(encoding="utf-8"))
        except Exception as e:
            console.print(f"[red]Error parsing YAML {f}: {e}[/red]")
            return []

        results = []
        loops_to_predict = []
        code_context = ""
        
        # Case 1: Invariant Result YAML
        if "invariants_result" in content:
            console.print(f"[blue]Detected Invariant Result YAML: {f.name}[/blue]")
            # For prediction, we usually look at the whole program or set of loops.
            # But 'predict' method takes 'code' and 'loops'.
            # If we have multiple loops in YAML, we can treat them as the set of loops for the program.
            
            # We need 'code' (full program). Invariant YAML might not have full code, just loop snippets.
            # However, 'predict' prompt uses 'code' and 'loops'.
            # If we only have snippets, we can pass the first snippet as code, or join them.
            # Or we can run prediction PER LOOP?
            # The 'prediction' prompt asks "decide whether the target C code terminates".
            # Usually it expects the full code.
            # If YAML only has snippets, we might be limited.
            # But let's assume we want to predict for each loop individually if they are isolated?
            # Or aggregate?
            # The 'predict' method signature: predict(code, loops, references, invariants, ranking_function)
            
            # Let's try to run prediction for each loop entry in the YAML.
            for item in content["invariants_result"]:
                loop_id = item.get("loop_id") or item.get("id")
                code = item.get("code", "")
                invs = item.get("invariants", [])
                
                console.print(f"  Predicting for Loop {loop_id}...")
                # We pass the loop code as 'code', and also as the single item in 'loops' list
                prediction, _ = predictor.predict(code, [code], references, invariants=invs)
                
                results.append({
                    "loop_id": loop_id,
                    "code": code,
                    "prediction": prediction
                })

        # Case 2: Extract Result YAML
        elif "loops" in content:
            console.print(f"[blue]Detected Extract Result YAML: {f.name}[/blue]")
            for item in content["loops"]:
                loop_id = item.get("id")
                code = item.get("code", "")
                
                console.print(f"  Predicting for Loop {loop_id}...")
                prediction, _ = predictor.predict(code, [code], references)
                
                results.append({
                    "loop_id": loop_id,
                    "code": code,
                    "prediction": prediction
                })
        else:
            console.print(f"[yellow]Unknown YAML format in {f.name}.[/yellow]")
            
        return results

    def process_c_file(f: Path) -> Dict[str, Any]:
        code = f.read_text(encoding="utf-8")
        loops = loop_extractor.extract(code)
        prediction, _ = predictor.predict(code, loops, references)
        return {
            "file": f.name,
            "prediction": prediction
        }

    if input.is_file():
        if input.suffix.lower() in {'.yml', '.yaml'}:
            results = process_yaml_input(input)
            if output:
                if output.is_dir():
                    out_path = output / (input.stem + "_prediction.yml")
                else:
                    out_path = output
                
                out_data = {
                    "source_file": input.name,
                    "task": "prediction",
                    "results": results
                }
                with open(out_path, 'w', encoding='utf-8') as f:
                    yaml.dump(out_data, f, sort_keys=False, allow_unicode=True)
                console.print(f"Saved prediction results to {out_path}")
            else:
                console.print(yaml.dump(results, sort_keys=False, allow_unicode=True))
        else:
            console.print(f"Predicting for {input}...")
            result = process_c_file(input)
            if output:
                if output.is_dir():
                    out_path = output / (input.stem + "_prediction.json")
                else:
                    out_path = output
                out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
                console.print(f"Saved to {out_path}")
            else:
                console.print(json.dumps(result, indent=2))

    elif input.is_dir():
        files = collect_files(input, recursive)
        
        ensure_output_dir(output)
            
        for f in files:
            try:
                if f.suffix.lower() in {'.yml', '.yaml'}:
                    results = process_yaml_input(f)
                    # Save YAML
                    out_path = output / (f.stem + "_prediction.yml")
                    out_data = {"source_file": f.name, "results": results}
                    with open(out_path, 'w', encoding='utf-8') as yf:
                        yaml.dump(out_data, yf, sort_keys=False, allow_unicode=True)
                else:
                    result = process_c_file(f)
                    out_path = output / (f.stem + "_prediction.json")
                    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
            except Exception as e:
                console.print(f"[red]Error processing {f.name}: {e}[/red]")


@app.command()
def z3verify(
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
                console.print(f"[red]Error verifying {f.name}: {e}[/red]")


@app.command()
def svmranker(
    input: Path = typer.Option(..., exists=True, help="Input YAML file or directory from ranking template output"),
    svm_ranker_path: Path = typer.Option(..., "--svm-ranker", help="Path to SVMRanker root directory"),
    output: Optional[Path] = typer.Option(None, help="Output file or directory"),
    recursive: bool = typer.Option(False, "--recursive", "-r", help="Recursively search for files if input is directory"),
) -> None:
    """
    Run SVMRanker using template parameters from ranking-template YAML output.
    """
    client = SVMRankerClient(str(svm_ranker_path))

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
                candidate = base_dir / source_path
                if candidate.exists():
                    path = candidate
            if path.exists():
                return path.read_text(encoding="utf-8")
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
            status, rf, rf_list = client.run(Path(tmp_path), mode=mode, depth=depth_val)
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

    def process_yaml(f: Path) -> Dict[str, Any]:
        try:
            content = yaml.safe_load(f.read_text(encoding="utf-8"))
        except Exception as e:
            return {"source_file": f.name, "error": f"YAML parse error: {e}"}

        entries = parse_results(content)
        base_dir = f.parent
        results = [run_on_entry(entry, base_dir) for entry in entries]
        return {"source_file": f.name, "task": "svmranker", "results": results}

    if input.is_file():
        result = process_yaml(input)
        if output:
            if output.is_dir():
                out_path = output / (input.stem + "_svmranker.yml")
            else:
                out_path = output
            with open(out_path, "w", encoding="utf-8") as yf:
                yaml.dump(result, yf, sort_keys=False, allow_unicode=True)
            console.print(f"Saved to {out_path}")
        else:
            console.print(yaml.dump(result, sort_keys=False, allow_unicode=True))
    elif input.is_dir():
        files = collect_files(input, recursive, extensions={".yml", ".yaml"})

        ensure_output_dir(output)

        for f in files:
            try:
                result = process_yaml(f)
                if output:
                    try:
                        rel_path = f.relative_to(input)
                        out_path = output / rel_path.with_suffix(".yml")
                        out_path = out_path.parent / (out_path.stem + "_svmranker.yml")
                    except ValueError:
                        out_path = output / (f.stem + "_svmranker.yml")
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(out_path, "w", encoding="utf-8") as yf:
                        yaml.dump(result, yf, sort_keys=False, allow_unicode=True)
                else:
                    console.print(f"--- {f.name} ---")
                    console.print(yaml.dump(result, sort_keys=False, allow_unicode=True))
            except Exception as e:
                console.print(f"[red]Error processing {f.name}: {e}[/red]")
