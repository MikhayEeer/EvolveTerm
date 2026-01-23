from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import json
import typer
import yaml
from rich.console import Console

from evolve_term.cli_utils import collect_files, ensure_output_dir
from evolve_term.verifiers.seahorn_verifier import SeaHornVerifier, DEFAULT_SEAHORN_IMAGE

console = Console()


def _parse_ranking_content(content: Any) -> Dict[int, str]:
    rankings: Dict[int, str] = {}
    if isinstance(content, str):
        rankings[1] = content.strip()
        return rankings
    if isinstance(content, list):
        for idx, entry in enumerate(content, start=1):
            if isinstance(entry, dict) and "ranking_function" in entry:
                loop_id = entry.get("loop_id") or entry.get("id") or idx
                rankings[int(loop_id)] = str(entry.get("ranking_function"))
            elif isinstance(entry, str) and idx == 1:
                rankings[1] = entry.strip()
        return rankings
    if isinstance(content, dict):
        if "ranking_function" in content:
            rankings[1] = str(content.get("ranking_function"))
        elif "ranking_results" in content:
            for idx, entry in enumerate(content.get("ranking_results") or [], start=1):
                if not isinstance(entry, dict):
                    continue
                loop_id = entry.get("loop_id") or entry.get("id") or idx
                if "ranking_function" in entry:
                    rankings[int(loop_id)] = str(entry.get("ranking_function"))
        return rankings
    return rankings


def _parse_invariants_content(content: Any) -> Dict[int, List[str]]:
    invariants: Dict[int, List[str]] = {}
    if isinstance(content, list):
        if all(isinstance(item, str) for item in content):
            invariants[1] = [str(item) for item in content]
        else:
            for idx, entry in enumerate(content, start=1):
                if not isinstance(entry, dict):
                    continue
                loop_id = entry.get("loop_id") or entry.get("id") or idx
                invs = entry.get("invariants") or []
                if isinstance(invs, list):
                    invariants[int(loop_id)] = [str(item) for item in invs]
        return invariants
    if isinstance(content, dict):
        if "invariants" in content and isinstance(content["invariants"], list):
            invariants[1] = [str(item) for item in content["invariants"]]
        elif "invariants_result" in content:
            for idx, entry in enumerate(content.get("invariants_result") or [], start=1):
                if not isinstance(entry, dict):
                    continue
                loop_id = entry.get("loop_id") or entry.get("id") or idx
                invs = entry.get("invariants") or []
                if isinstance(invs, list):
                    invariants[int(loop_id)] = [str(item) for item in invs]
        elif "ranking_results" in content:
            for idx, entry in enumerate(content.get("ranking_results") or [], start=1):
                if not isinstance(entry, dict):
                    continue
                loop_id = entry.get("loop_id") or entry.get("id") or idx
                invs = entry.get("invariants") or []
                if isinstance(invs, list):
                    invariants[int(loop_id)] = [str(item) for item in invs]
        return invariants
    return invariants


def _load_structured_file(path: Path) -> Tuple[Optional[Any], Optional[str]]:
    try:
        if path.suffix.lower() == ".json":
            return json.loads(path.read_text(encoding="utf-8")), None
        if path.suffix.lower() in {".yml", ".yaml"}:
            return yaml.safe_load(path.read_text(encoding="utf-8")), None
        return path.read_text(encoding="utf-8").strip(), None
    except Exception as exc:
        return None, str(exc)


class SeaVerifyHandler:
    def __init__(self, docker_image: str = DEFAULT_SEAHORN_IMAGE, timeout_seconds: int = 60, verbose: bool = False):
        self.verifier = SeaHornVerifier(docker_image=docker_image, timeout_seconds=timeout_seconds)
        self.verbose = verbose

    def run(
        self,
        input: Path,
        ranking_func: Optional[str],
        ranking_file: Optional[Path],
        invariants_file: Optional[Path],
        output: Optional[Path],
        recursive: bool,
    ) -> None:
        console.print(f"[cyan]SeaHorn image: {self.verifier.docker_image}[/cyan]")
        console.print(f"[cyan]SeaHorn timeout: {self.verifier.timeout_seconds}s[/cyan]")

        def get_rankings(f_path: Path) -> Tuple[Dict[int, str], Optional[str]]:
            if ranking_func:
                return {1: ranking_func}, "inline"
            if not ranking_file:
                return {}, None
            if ranking_file.is_dir():
                candidates = [
                    ranking_file / (f_path.stem + ext)
                    for ext in (".json", ".yml", ".yaml", ".txt")
                ]
                candidate = next((p for p in candidates if p.exists()), None)
                if not candidate:
                    return {}, None
                content, err = _load_structured_file(candidate)
                if err:
                    console.print(f"[yellow]Failed to read ranking file {candidate}: {err}[/yellow]")
                    return {}, str(candidate)
                return _parse_ranking_content(content), str(candidate)
            if ranking_file.exists():
                content, err = _load_structured_file(ranking_file)
                if err:
                    console.print(f"[yellow]Failed to read ranking file {ranking_file}: {err}[/yellow]")
                    return {}, str(ranking_file)
                return _parse_ranking_content(content), str(ranking_file)
            return {}, None

        def get_invariants(f_path: Path) -> Tuple[Dict[int, List[str]], Optional[str]]:
            if not invariants_file and not ranking_file:
                return {}, None
            target = invariants_file
            if target and target.is_dir():
                candidates = [
                    target / (f_path.stem + ext)
                    for ext in (".json", ".yml", ".yaml")
                ]
                candidate = next((p for p in candidates if p.exists()), None)
                if not candidate:
                    return {}, None
                content, err = _load_structured_file(candidate)
                if err:
                    console.print(f"[yellow]Failed to read invariants file {candidate}: {err}[/yellow]")
                    return {}, str(candidate)
                return _parse_invariants_content(content), str(candidate)
            if target and target.exists():
                content, err = _load_structured_file(target)
                if err:
                    console.print(f"[yellow]Failed to read invariants file {target}: {err}[/yellow]")
                    return {}, str(target)
                return _parse_invariants_content(content), str(target)

            if ranking_file:
                if ranking_file.is_dir():
                    candidates = [
                        ranking_file / (f_path.stem + ext)
                        for ext in (".json", ".yml", ".yaml", ".txt")
                    ]
                    candidate = next((p for p in candidates if p.exists()), None)
                    if not candidate:
                        return {}, None
                    content, err = _load_structured_file(candidate)
                    if err:
                        console.print(f"[yellow]Failed to read ranking file {candidate}: {err}[/yellow]")
                        return {}, str(candidate)
                    return _parse_invariants_content(content), str(candidate)
                content, err = _load_structured_file(ranking_file)
                if err:
                    console.print(f"[yellow]Failed to read ranking file {ranking_file}: {err}[/yellow]")
                    return {}, str(ranking_file)
                return _parse_invariants_content(content), str(ranking_file)
            return {}, None

        def process_file(f: Path) -> str:
            code = f.read_text(encoding="utf-8")
            loop_ranking, ranking_source = get_rankings(f)
            loop_invs, invariants_source = get_invariants(f)
            self._print_sources(f.name, ranking_source, invariants_source, loop_ranking, loop_invs)
            result = self.verifier.verify(code, loop_invs, loop_ranking)
            self._print_report(f.name, self.verifier.last_report)
            self._print_run_details(result)
            return result

        if input.is_file():
            console.print(f"SeaHorn verifying {input}...")
            result = process_file(input)
            self._write_output(input, output, result)
        elif input.is_dir():
            files = collect_files(input, recursive)
            ensure_output_dir(output)
            for f in files:
                try:
                    console.print(f"SeaHorn verifying {f.name}...")
                    result = process_file(f)
                    self._write_output(f, output, result, base_dir=input)
                except Exception as exc:
                    console.print(f"[red]Error verifying {f.name}: {exc}[/red]")

    def _write_output(self, input_path: Path, output: Optional[Path], result: str, base_dir: Optional[Path] = None) -> None:
        if output:
            if output.is_dir():
                if base_dir:
                    try:
                        rel_path = input_path.relative_to(base_dir)
                        out_path = output / rel_path.with_suffix(".txt")
                    except ValueError:
                        out_path = output / (input_path.stem + ".txt")
                else:
                    out_path = output / (input_path.stem + ".txt")
            else:
                out_path = output
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(result, encoding="utf-8")
            console.print(f"Saved to {out_path}")
        else:
            console.print(f"Result: {result}")

    def _print_report(self, name: str, report: Optional[Dict[str, object]]) -> None:
        if not report:
            return
        skipped = report.get("skipped_loops") or []
        dropped_invs = report.get("dropped_invariants") or []
        dropped_rankings = report.get("dropped_rankings") or []
        instrumented = report.get("instrumented_loop_ids") or []
        total_loops = report.get("total_loops")

        if total_loops is not None:
            console.print(
                f"[cyan]{name}: total loops={total_loops}, instrumented={len(instrumented)}[/cyan]"
            )

        if skipped:
            console.print(f"[yellow]{name}: skipped {len(skipped)} loop(s) due to parsing limitations.[/yellow]")
        if dropped_invs:
            console.print(f"[yellow]{name}: dropped {len(dropped_invs)} non-C invariant(s).[/yellow]")
        if dropped_rankings:
            console.print(f"[yellow]{name}: dropped {len(dropped_rankings)} non-C ranking function(s).[/yellow]")
        if not instrumented and (dropped_invs or dropped_rankings):
            console.print(f"[yellow]{name}: no loop instrumentation applied after filtering.[/yellow]")

    def _print_sources(
        self,
        name: str,
        ranking_source: Optional[str],
        invariants_source: Optional[str],
        loop_ranking: Dict[int, str],
        loop_invs: Dict[int, List[str]],
    ) -> None:
        ranking_note = ranking_source or "none"
        inv_note = invariants_source or "none"
        console.print(
            f"[cyan]{name}: ranking source={ranking_note} (loops={len(loop_ranking)})[/cyan]"
        )
        console.print(
            f"[cyan]{name}: invariants source={inv_note} (loops={len(loop_invs)})[/cyan]"
        )

    def _print_run_details(self, result: str) -> None:
        if not self.verbose or not self.verifier.last_run:
            return
        cmd = self.verifier.last_run.get("command") or []
        if cmd:
            console.print(f"[cyan]SeaHorn cmd: {' '.join(cmd)}[/cyan]")
        stdout = (self.verifier.last_run.get("stdout") or "").strip()
        stderr = (self.verifier.last_run.get("stderr") or "").strip()
        if result.startswith("Error") or result.startswith("Failed"):
            if stdout:
                console.print(f"[yellow]SeaHorn stdout: {stdout[:500]}[/yellow]")
            if stderr:
                console.print(f"[yellow]SeaHorn stderr: {stderr[:500]}[/yellow]")
