"""Typer-based CLI to interact with the termination pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Optional
import json

import typer
from rich.console import Console
from rich.table import Table


from .models import PendingReviewCase
from .pipeline import TerminationPipeline
from .cli_utils import (
    resolve_svm_ranker_root,
    ping_llm_client,
    DEFAULT_LLM_PING_PROMPT,
    check_loop_id_order_in_yaml,
    collect_files,
    validate_yaml_required_keys,
    ValidationResult,
)
from .yaml_schema import validate_yaml_file as validate_yaml_file_full

# Handlers
from .commands.extract import ExtractHandler
from .commands.invariant import InvariantHandler
from .commands.ranking import RankingHandler
from .commands.predict import PredictHandler
from .commands.svmranker import SVMRankerHandler
from .commands.z3verify import Z3VerifyHandler
from .commands.sea_verify import SeaVerifyHandler
from .commands.batch import BatchHandler
from .commands.translate import TranslateHandler
from .commands.feature import FeatureHandler

app = typer.Typer(help="EvolveTerm CLI - analyze and curate C termination cases")
console = Console()
VALID_LABELS = {"terminating", "non-terminating", "unknown"}
SVM_RANKER_EXAMPLE = "evolveterm analyze --code-file examples/llm_nested_term_samples/generated1229program1/program.c --svm-ranker /path/to/SVMRanker"
SVM_RANKER_HELP = (
    "SVMRanker 仓库根目录（包含 src/CLIMain.py）。"
    " 若传入的是 src 目录或 CLIMain.py 文件，会自动纠正到根目录。"
)


@app.command()
def analyze(
    code_file: Path = typer.Option(..., exists=True, readable=True, help="Path to a source file"),
    top_k: int = 5,
    enable_translation: bool = typer.Option(False, "--enable-translation", "-t", help="Enable LLM-based translation to C++ for non-C/C++ files"),
    knowledge_base: Optional[Path] = typer.Option(None, "--kb", help="Path to a custom knowledge base JSON file"),
    use_rag_reasoning: bool = typer.Option(True, "--use-rag-reasoning/--no-rag-reasoning", 
                                           help="Use RAG references for invariant and ranking function inference; Default is enabled"),
    svm_ranker_path: Optional[Path] = typer.Option(None, "--svm-ranker", "--svmranker", help=SVM_RANKER_HELP),
    known_terminating: bool = typer.Option(False, "--known-terminating", help="Hint that the program is known to terminate"),
    verifier_backend: str = typer.Option("z3", "--verifier", help="Verification backend: z3 or seahorn"),
    seahorn_image: str = typer.Option("seahorn/seahorn-llvm14:nightly", "--seahorn-image", help="SeaHorn docker image"),
    seahorn_timeout: int = typer.Option(60, "--seahorn-timeout", help="SeaHorn timeout seconds"),
    # Ablation parameters
    extraction_prompt_version: str = typer.Option("v2", "--prompt-version", "-p", help="Prompt version for loop extraction (v1 or v2)"),
    use_loops_for_embedding: bool = typer.Option(True, "--embed-loops/--embed-code", help="Use extracted loops for embedding vs full code"),
    use_loops_for_reasoning: bool = typer.Option(True, "--reason-loops/--reason-code", help="Use extracted loops for reasoning vs full code")
) -> None:
    """分析 C/C++ 代码的终止性。

    示例（可直接复制）:
        {example}
    """.format(example=SVM_RANKER_EXAMPLE)

    # Check file extension
    suffix = code_file.suffix.lower()
    is_cpp = suffix in {".c", ".cpp", ".h", ".hpp", ".cc", ".cxx"}
    
    if not is_cpp and not enable_translation:
        console.print(f"[white on red]FATAL ERROR[/white on red] File '{code_file.name}' does not appear to be a C/C++ file.")
        console.print("[bright_cyan]INFO:[/bright_cyan] Please use [bold]--enable-translation[/bold] to enable automatic translation.")
        raise typer.Exit(code=1)

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
    code = code_file.read_text(encoding="utf-8")
    result = pipeline.analyze(
        code, 
        top_k=top_k, 
        use_rag_in_reasoning=use_rag_reasoning,
        use_svm_ranker=bool(svm_ranker_root),
        known_terminating=known_terminating,
        extraction_prompt_version=extraction_prompt_version,
        use_loops_for_embedding=use_loops_for_embedding,
        use_loops_for_reasoning=use_loops_for_reasoning
    )

    # Show translation info if applicable
    if enable_translation and result.report_path:
        with open(result.report_path, 'r', encoding='utf-8') as f:
            report = json.load(f)
        translation = report.get("translation", {})
        if translation.get("translated"):
            console.print("[bright_green]SUCCESS: Code was translated to C++[/bright_green]")

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
    svm_ranker_path: Optional[Path] = typer.Option(None, "--svm-ranker", "--svmranker", help=SVM_RANKER_HELP),
    known_terminating: bool = typer.Option(False, "--known-terminating", help="Hint that the program is known to terminate"),
    verifier_backend: str = typer.Option("z3", "--verifier", help="Verification backend: z3 or seahorn"),
    seahorn_image: str = typer.Option("seahorn/seahorn-llvm14:nightly", "--seahorn-image", help="SeaHorn docker image"),
    seahorn_timeout: int = typer.Option(60, "--seahorn-timeout", help="SeaHorn timeout seconds"),
    # Ablation parameters
    extraction_prompt_version: str = typer.Option("v2", "--prompt-version", "-p", help="Prompt version for loop extraction (v1 or v2)"),
    use_loops_for_embedding: bool = typer.Option(True, "--embed-loops/--embed-code", help="Use extracted loops for embedding vs full code"),
    use_loops_for_reasoning: bool = typer.Option(True, "--reason-loops/--reason-code", help="Use extracted loops for reasoning vs full code")
) -> None:
    """Batch analyze all C/C++ files in a directory."""
    handler = BatchHandler()
    handler.run(
        input_dir, top_k, enable_translation, knowledge_base, recursive, 
        use_rag_reasoning, svm_ranker_path, known_terminating, 
        extraction_prompt_version, use_loops_for_embedding, use_loops_for_reasoning,
        verifier_backend, seahorn_image, seahorn_timeout
    )


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
    handler = TranslateHandler(llm_config)
    handler.run(input, output, recursive)


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
        Filename format: {name}_{prompt_version}_extract.yml
    """
    handler = ExtractHandler(llm_config)
    handler.run(input, output, recursive, prompt_version)



@app.command()
def invariant(
    input: Path = typer.Option(..., exists=True, help="Input code file, YAML file, or directory"),
    references_file: Optional[Path] = typer.Option(None, help="JSON or YAML file containing reference cases (List[KnowledgeCase dict])"),
    output: Optional[Path] = typer.Option(None, help="Output file or directory"),
    llm_config: str = typer.Option("llm_config.json", help="Path to LLM config"),
    recursive: bool = typer.Option(False, "--recursive", "-r", help="Recursively search for files if input is directory"),
    mode: str = typer.Option("auto", "--mode", "-m", help="Filter mode for batch processing: 'auto' (all), 'yaml' (only extract results), 'code' (only C/C++ files)"),
    extract_prompt_version: str = typer.Option("all", "--extract-pmt-v", "--extv", help="Filter YAML files by extract prompt version in filename: all, v1, v2"),
    prompt_version: str = typer.Option(
        "acsl_cot",
        "--prompt-version",
        "-pv",
        help=(
            "Prompt version for invariant inference: "
            "'acsl_cot' (default), 'acsl_direct', 'acsl_cot_fewshot', "
            "'seahorn_cot', 'seahorn_direct', or 'seahorn_cot_fewshot'"
        ),
    ),
    fill_empty_invariants: bool = typer.Option(False, "--fill-empty-invariants", help="For invariant-result YAMLs, re-infer only empty invariants (invariants: [])"),
) -> None:
    """
    Infer invariants for the given code or extracted YAML.
    
    Input can be:
    1. A C/C++ source file.
    2. A YAML file generated by the 'extract' command.
    3. A directory containing such files.

    References JSON/YAML Format:
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
    handler = InvariantHandler(llm_config)
    handler.run(input, references_file, output, recursive, mode, extract_prompt_version, prompt_version, fill_empty_invariants)


@app.command()
def ranking(
    input: Path = typer.Option(..., exists=True, help="Input code file, YAML file, or directory"),
    invariants_file: Optional[Path] = typer.Option(None, help="JSON or YAML file containing invariants list (for single file input)"),
    references_file: Optional[Path] = typer.Option(None, help="JSON or YAML file containing reference cases"),
    output: Optional[Path] = typer.Option(None, help="Output file or directory"),
    llm_config: str = typer.Option("llm_config.json", help="Path to LLM config"),
    recursive: bool = typer.Option(False, "--recursive", "-r", help="Recursively search for files if input is directory"),
    mode: str = typer.Option("auto", "--mode", "-m", help="Filter mode for batch processing: 'auto' (all), 'yaml' (only extract/invariant results), 'code' (only C/C++ files)"),
    ranking_mode: str = typer.Option(
        "template",
        "--ranking-mode",
        help="Ranking prompt mode: 'direct', 'template' (default), 'template-fewshot', 'template-known', or 'template-known-fewshot'",
    ),
    retry_empty: int = typer.Option(2, "--retry-empty", help="Max retries when ranking result is empty"),
) -> None:
    """
    Infer ranking function for the given code.
    
    Input can be:
    1. A C/C++ source file.
    2. A YAML file generated by 'extract' or 'invariant' command.
    3. A directory containing such files.
    
    """
    handler = RankingHandler(llm_config)
    handler.run(input, invariants_file, references_file, output, recursive, mode, ranking_mode, retry_empty)


@app.command()
def predict(
    input: Path = typer.Option(..., exists=True, help="Input code file, directory, or YAML"),
    references_file: Optional[Path] = typer.Option(None, help="JSON or YAML file containing reference cases"),
    llm_config: str = typer.Option("llm_config.json", help="Path to LLM config"),
    output: Optional[Path] = typer.Option(None, help="Output file or directory"),
    recursive: bool = typer.Option(False, "--recursive", "-r", help="Recursively search for files if input is directory"),
) -> None:
    """
    Perform preliminary termination prediction.
    Supports C files (auto-extract loops) or YAML files (from extract/invariant modules).
    """
    handler = PredictHandler(llm_config)
    handler.run(input, references_file, output, recursive)


@app.command()
def z3verify(
    input: Path = typer.Option(..., exists=True, help="Input code file or directory"),
    ranking_func: Optional[str] = typer.Option(None, help="Ranking function string (for single file)"),
    ranking_file: Optional[Path] = typer.Option(None, help="File containing ranking function (JSON/YAML {ranking_function: ...} or raw text)"),
    invariants_file: Optional[Path] = typer.Option(None, help="JSON or YAML file containing invariants list"),
    output: Optional[Path] = typer.Option(None, help="Output file or directory"),
    llm_config: str = typer.Option("llm_config.json", help="Path to LLM config"),
    recursive: bool = typer.Option(False, "--recursive", "-r", help="Recursively search for files if input is directory"),
) -> None:
    """
    Verify ranking function and invariants using Z3.
    """
    handler = Z3VerifyHandler(llm_config)
    handler.run(input, ranking_func, ranking_file, invariants_file, output, recursive)


@app.command()
def seaverify(
    input: Path = typer.Option(..., exists=True, help="Input code file or directory"),
    ranking_func: Optional[str] = typer.Option(None, help="Ranking function string (for single file)"),
    ranking_file: Optional[Path] = typer.Option(None, help="File containing ranking function (JSON/YAML or raw text)"),
    invariants_file: Optional[Path] = typer.Option(None, help="File containing invariants (JSON/YAML)"),
    output: Optional[Path] = typer.Option(None, help="Output file or directory"),
    docker_image: str = typer.Option("seahorn/seahorn-llvm14:nightly", "--docker-image", help="SeaHorn docker image"),
    timeout: int = typer.Option(60, "--timeout", help="Timeout seconds for SeaHorn"),
    verbose: bool = typer.Option(False, "--verbose", help="Print SeaHorn command and diagnostics"),
    recursive: bool = typer.Option(False, "--recursive", "-r", help="Recursively search for files if input is directory"),
) -> None:
    """
    Verify ranking function and invariants using SeaHorn (Docker).
    """
    handler = SeaVerifyHandler(docker_image=docker_image, timeout_seconds=timeout, verbose=verbose)
    handler.run(input, ranking_func, ranking_file, invariants_file, output, recursive)


@app.command()
def svmranker(
    input: Path = typer.Option(..., exists=True, help="Input YAML file or directory from ranking template output"),
    svm_ranker_path: Path = typer.Option(..., "--svm-ranker", "--svmranker", help=SVM_RANKER_HELP),
    output: Path = typer.Option(..., file_okay=False, dir_okay=True, help="Output directory"),
    recursive: bool = typer.Option(False, "--recursive", "-r", help="Recursively search for files if input is directory"),
    rerun_failed: bool = typer.Option(
        False,
        "--rerun-failed",
        help="Rerun failed SVMRanker outputs from an existing failed directory.",
    ),
) -> None:
    """
    Run SVMRanker using template parameters from ranking-template YAML output.
    Use --rerun-failed to rerun failed SVMRanker outputs.
    """
    handler = SVMRankerHandler(svm_ranker_path)
    if rerun_failed:
        handler.rerun_failed(input, output, recursive)
    else:
        handler.run(input, output, recursive)


@app.command()
def feature(
    input: Path = typer.Option(..., exists=True, help="Input file or directory (C/C++)"),
    output: Optional[Path] = typer.Option(None, help="Output directory"),
    llm_config: str = typer.Option("llm_config.json", help="Path to LLM config"),
    recursive: bool = typer.Option(False, "--recursive", "-r", help="Recursively search for files if input is directory"),
) -> None:
    """
    Extract code features (loop count, depth, semantic summary, etc.) for code summarization.
    """
    handler = FeatureHandler(llm_config)
    handler.run(input, output, recursive)


@app.command()
def validate_yaml(
    input: Path = typer.Option(..., exists=True, help="YAML file or directory to validate"),
    recursive: bool = typer.Option(False, "--recursive", "-r", help="Recursively search for YAML files"),
    strict: bool = typer.Option(False, "--strict", help="Treat warnings as errors"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed validation results"),
    #showSuccess: bool = typer.Option(False, "--show-success", help="Also print yaml names witchs are successfully passed check.")
) -> None:
    """Validate YAML files against EvolveTerm schemas.
    
    Checks:
    - Required fields presence
    - Field types
    - Loop ID sequentiality (1..N)
    - Loop nesting rules (LOOP{n} references only previous loops)
    """
    files: List[Path] = []
    
    if input.is_file():
        files = [input]
    else:
        files = collect_files(input, recursive, extensions={".yml", ".yaml"})
    
    if not files:
        console.print("[yellow]No YAML files found.[/yellow]")
        return
    
    console.print(f"Validating {len(files)} YAML file(s)...\n")
    
    valid_count = 0
    invalid_count = 0
    warning_count = 0
    
    for path in files:
        result = validate_yaml_file_full(path, strict=strict)
        
        if result.valid:
            valid_count += 1
            if verbose:
                console.print(f"[bright_green]SUCCESS: {path.name}[/bright_green]")
                if result.warnings:
                    for warn in result.warnings:
                        console.print(f"  [black on yellow]WARNING[/black on yellow] {warn}")
        else:
            invalid_count += 1
            console.print(f"[bold red]ERROR: {path.name}[/bold red]")
            for err in result.errors:
                console.print(f"  [bold red]ERROR:[/bold red] {err}")
            if result.warnings:
                warning_count += len(result.warnings)
                for warn in result.warnings:
                    console.print(f"  [black on yellow]WARNING[/black on yellow] {warn}")
    
    console.print()
    console.print(f"Validation Summary:")
    console.print(f"  Valid:   [bright_green]{valid_count}[/bright_green]")
    console.print(f"  Invalid: [bold red]{invalid_count}[/bold red]")
    if warning_count > 0:
        console.print(f"  Warnings: [black on yellow]{warning_count}[/black on yellow]")
    
    if invalid_count > 0:
        raise typer.Exit(code=1)


def _run_llm_ping(llm_config: str, prompt: str) -> None:
    try:
        result = ping_llm_client(llm_config, prompt)
    except Exception as exc:
        console.print(f"[white on red]FATAL ERROR[/white on red] Failed to connect to LLM API: {exc}")
        raise typer.Exit(code=1)

    console.rule("LLM Ping")
    console.print("Config tag: default")
    console.print("LLM config:")
    console.print(json.dumps(result["config"], ensure_ascii=False, indent=2), markup=False)
    console.print("Prompt:")
    console.print(result["prompt"], markup=False)
    console.print("Response:")
    console.print(result["response"], markup=False)


@app.command("ping-llm")
def ping_llm(
    llm_config: str = typer.Option("llm_config.json", help="Path to LLM config"),
    prompt: str = typer.Option(DEFAULT_LLM_PING_PROMPT, "--prompt", help="Short prompt for ping test"),
) -> None:
    """Ping LLM API with default tag."""
    _run_llm_ping(llm_config, prompt)
