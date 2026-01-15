#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import csv
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

CURRENT_FILE = Path(__file__).resolve()
SRC_ROOT = CURRENT_FILE.parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from evolve_term.sea_verify import SeaHornVerifier, DEFAULT_SEAHORN_IMAGE


DEFAULT_OUTPUT_ROOT = Path("src/prompts_playground/output")
DEFAULT_DATA_ROOT = Path("data")
DEFAULT_GLOB_PATTERN = "Loo*/**/*.csv"


@dataclass
class RunResult:
    status: str
    reason: str
    command: List[str]
    returncode: Optional[int]
    stdout: str
    stderr: str


def slugify(value: str, max_len: int = 80) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    cleaned = cleaned.strip("_") or "item"
    return cleaned[:max_len]


def parse_invariants_field(raw_value: Optional[str]) -> Tuple[List[str], Optional[str]]:
    if raw_value is None:
        return [], None
    text = str(raw_value).strip()
    if not text:
        return [], None
    parsers = (json.loads, ast.literal_eval)
    last_error = None
    for parser in parsers:
        try:
            parsed = parser(text)
        except Exception as exc:
            last_error = str(exc)
            continue
        if isinstance(parsed, list):
            return [str(item).strip() for item in parsed if str(item).strip()], None
        if isinstance(parsed, str):
            return [parsed.strip()], None
    return [], f"parse_failed: {last_error or 'unknown'}"


def discover_csv_files(root: Path, pattern: str) -> List[Path]:
    if root.is_file():
        return [root]
    return sorted(root.glob(pattern))


def resolve_source_dir(csv_path: Path, output_root: Path, data_root: Path) -> Optional[Path]:
    try:
        rel = csv_path.relative_to(output_root)
    except ValueError:
        return None
    return data_root / rel.parent


def run_seahorn(
    code_path: Path,
    docker_image: str,
    timeout_seconds: int,
    docker_name: Optional[str] = None,
) -> RunResult:
    cmd = ["docker", "run", "--rm"]
    if docker_name:
        cmd += ["--name", docker_name]
    cmd += [
        "-v",
        f"{code_path.parent.resolve()}:/src",
        docker_image,
        "sea",
        "pf",
        f"/src/{code_path.name}",
    ]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
        stdout = result.stdout or ""
        stderr = result.stderr or ""
        combined = f"{stdout}\n{stderr}".strip()
        status = "error"
        reason = "unexpected_output"
        if re.search(r"\bunsat\b", combined, re.IGNORECASE):
            status = "Verified"
            reason = ""
        elif re.search(r"\bsat\b", combined, re.IGNORECASE):
            status = "Failed"
            reason = ""
        elif result.returncode != 0:
            reason = f"exit_{result.returncode}"
        return RunResult(
            status=status,
            reason=reason,
            command=cmd,
            returncode=result.returncode,
            stdout=stdout,
            stderr=stderr,
        )
    except FileNotFoundError:
        return RunResult(
            status="error",
            reason="docker_not_found",
            command=cmd,
            returncode=None,
            stdout="",
            stderr="docker: command not found",
        )
    except subprocess.TimeoutExpired as exc:
        stdout = getattr(exc, "stdout", "") or ""
        stderr = getattr(exc, "stderr", "") or ""
        return RunResult(
            status="error",
            reason="timeout",
            command=cmd,
            returncode=None,
            stdout=stdout,
            stderr=stderr or "docker: timeout",
        )
    except Exception as exc:
        return RunResult(
            status="error",
            reason="exception",
            command=cmd,
            returncode=None,
            stdout="",
            stderr=str(exc) or "docker: unknown error",
        )


def write_log(
    log_path: Path,
    header: Dict[str, Any],
    invariants_raw: List[str],
    invariants_cleaned: List[str],
    invariants_dropped: List[Dict[str, str]],
    scan_report: Dict[str, Any],
    instrument_report: Dict[str, Any],
    run_result: RunResult,
) -> None:
    lines: List[str] = []
    lines.append("== Meta ==")
    for key, value in header.items():
        lines.append(f"{key}: {value}")

    lines.append("")
    lines.append("== Invariants ==")
    lines.append(f"raw_count: {len(invariants_raw)}")
    lines.append(f"cleaned_count: {len(invariants_cleaned)}")
    lines.append(f"dropped_count: {len(invariants_dropped)}")
    if invariants_raw:
        lines.append("raw:")
        for inv in invariants_raw:
            lines.append(f"- {inv}")
    if invariants_cleaned:
        lines.append("cleaned:")
        for inv in invariants_cleaned:
            lines.append(f"- {inv}")
    if invariants_dropped:
        lines.append("dropped:")
        for item in invariants_dropped:
            lines.append(f"- {item.get('invariant', '')} ({item.get('reason', '')})")

    lines.append("")
    lines.append("== Loop Scan ==")
    lines.append(json.dumps(scan_report, ensure_ascii=True, indent=2))

    lines.append("")
    lines.append("== Instrumentation ==")
    lines.append(json.dumps(instrument_report, ensure_ascii=True, indent=2))

    lines.append("")
    lines.append("== SeaHorn ==")
    lines.append(f"status: {run_result.status}")
    if run_result.reason:
        lines.append(f"reason: {run_result.reason}")
    if run_result.command:
        lines.append(f"command: {' '.join(run_result.command)}")
    lines.append(f"returncode: {run_result.returncode}")
    lines.append("-- stdout --")
    lines.append(run_result.stdout.strip() or "(empty)")
    lines.append("-- stderr --")
    lines.append(run_result.stderr.strip() or "(empty)")

    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify Loopy CSV invariants with SeaHorn and save logs.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="CSV file or directory containing prompt outputs (default: src/prompts_playground/output).",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default=DEFAULT_GLOB_PATTERN,
        help="Glob pattern under --input to find CSVs (default: Loo*/**/*.csv).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Root used to compute CSV relative paths (default: src/prompts_playground/output).",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="Root for source C files (default: data).",
    )
    parser.add_argument(
        "--logs-dir",
        type=Path,
        default=Path("results/seahorn_csv_logs"),
        help="Directory to store SeaHorn logs and summaries.",
    )
    parser.add_argument(
        "--docker-image",
        type=str,
        default=DEFAULT_SEAHORN_IMAGE,
        help=f"SeaHorn Docker image (default: {DEFAULT_SEAHORN_IMAGE}).",
    )
    parser.add_argument(
        "--docker-name",
        type=str,
        default="",
        help="Optional docker --name value (blank disables).",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="SeaHorn timeout in seconds (default: 60).",
    )
    parser.add_argument(
        "--loop-id",
        type=int,
        default=1,
        help="Loop id to instrument (default: 1).",
    )
    parser.add_argument(
        "--apply-to-all-loops",
        action="store_true",
        help="Apply the same invariant list to every loop in the file.",
    )
    parser.add_argument(
        "--keep-instrumented",
        action="store_true",
        help="Keep instrumented C files next to the logs.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Max number of CSV rows to process across all files (0 = no limit).",
    )
    args = parser.parse_args()

    input_root = args.input
    csv_paths = discover_csv_files(input_root, args.pattern)
    if not csv_paths:
        print(f"No CSV files found under {input_root} with pattern {args.pattern}.")
        return 1

    logs_dir = args.logs_dir
    logs_dir.mkdir(parents=True, exist_ok=True)
    summary_path = logs_dir / "summary.csv"

    verifier = SeaHornVerifier(docker_image=args.docker_image, timeout_seconds=args.timeout)
    docker_name = args.docker_name.strip() or None

    summary_rows: List[Dict[str, Any]] = []
    processed_rows = 0

    for csv_path in csv_paths:
        if args.limit and processed_rows >= args.limit:
            break
        try:
            rel_csv = csv_path.relative_to(args.output_root)
            csv_log_base = logs_dir / rel_csv.parent / csv_path.stem
        except ValueError:
            csv_log_base = logs_dir / csv_path.stem
        csv_log_base.mkdir(parents=True, exist_ok=True)

        source_dir = resolve_source_dir(csv_path, args.output_root, args.data_root)

        with open(csv_path, newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row_index, row in enumerate(reader, start=1):
                if args.limit and processed_rows >= args.limit:
                    break
                processed_rows += 1

                target_file = (row.get("target_file") or "").strip() or f"{csv_path.stem}"
                source_path = None
                if source_dir:
                    source_path = source_dir / target_file
                if not source_path or not source_path.exists():
                    source_path = args.data_root / target_file
                if not source_path.exists():
                    log_dir = csv_log_base / f"row_{row_index:04d}_missing_source"
                    log_dir.mkdir(parents=True, exist_ok=True)
                    log_path = log_dir / "seahorn.log"
                    header = {
                        "csv": str(csv_path),
                        "row_index": row_index,
                        "target_file": target_file,
                        "source_file": str(source_path),
                        "status": "error",
                        "reason": "missing_source",
                    }
                    write_log(
                        log_path,
                        header,
                        [],
                        [],
                        [],
                        {"total_loops": 0, "skipped_loops": []},
                        {"instrumented_loop_ids": []},
                        RunResult(
                            status="error",
                            reason="missing_source",
                            command=[],
                            returncode=None,
                            stdout="",
                            stderr="",
                        ),
                    )
                    summary_rows.append(
                        {
                            "csv": str(csv_path),
                            "row_index": row_index,
                            "target_file": target_file,
                            "source_file": str(source_path),
                            "status": "error",
                            "reason": "missing_source",
                            "log_file": str(log_path),
                        }
                    )
                    continue

                invariants_raw, parse_error = parse_invariants_field(row.get("parsed_invariants"))
                cleaned_invs, dropped_invs = verifier._sanitize_invariants(invariants_raw)

                code = source_path.read_text(encoding="utf-8")
                loops, scan_report = verifier.scan_loops(code)
                total_loops = scan_report.get("total_loops", 0) or 0

                invariants_by_loop: Dict[int, List[str]] = {}
                if args.apply_to_all_loops and total_loops > 0:
                    for loop in loops:
                        invariants_by_loop[loop.loop_id] = cleaned_invs
                elif total_loops >= args.loop_id:
                    invariants_by_loop[args.loop_id] = cleaned_invs

                instrumented_code, instrument_report = verifier._instrument_code(
                    code, invariants_by_loop, {}
                )

                exp_name = row.get("experiment_name") or "experiment"
                model_name = row.get("model") or "model"
                prompt_version = row.get("prompt_version") or "prompt"
                slug = slugify(f"{exp_name}_{prompt_version}_{model_name}")
                log_dir = csv_log_base / f"row_{row_index:04d}_{slug}"
                log_dir.mkdir(parents=True, exist_ok=True)
                log_path = log_dir / "seahorn.log"
                instrumented_path = log_dir / "instrumented.c"

                header = {
                    "csv": str(csv_path),
                    "row_index": row_index,
                    "target_file": target_file,
                    "source_file": str(source_path),
                    "experiment_name": exp_name,
                    "prompt_version": prompt_version,
                    "model": model_name,
                    "temperature": row.get("temperature", ""),
                    "top_p": row.get("top_p", ""),
                    "config_tag": row.get("config_tag", ""),
                    "parse_error": parse_error or "",
                    "loop_id": "all" if args.apply_to_all_loops else args.loop_id,
                }

                if not cleaned_invs:
                    header["status"] = "skipped"
                    header["reason"] = "no_valid_invariants"
                    write_log(
                        log_path,
                        header,
                        invariants_raw,
                        cleaned_invs,
                        [{"invariant": inv, "reason": reason} for inv, reason in dropped_invs],
                        scan_report,
                        instrument_report,
                        RunResult(
                            status="skipped",
                            reason="no_valid_invariants",
                            command=[],
                            returncode=None,
                            stdout="",
                            stderr="",
                        ),
                    )
                    summary_rows.append(
                        {
                            "csv": str(csv_path),
                            "row_index": row_index,
                            "target_file": target_file,
                            "source_file": str(source_path),
                            "status": "skipped",
                            "reason": "no_valid_invariants",
                            "log_file": str(log_path),
                        }
                    )
                    continue

                if not instrument_report.get("instrumented_loop_ids"):
                    header["status"] = "skipped"
                    header["reason"] = "no_loop_instrumented"
                    write_log(
                        log_path,
                        header,
                        invariants_raw,
                        cleaned_invs,
                        [{"invariant": inv, "reason": reason} for inv, reason in dropped_invs],
                        scan_report,
                        instrument_report,
                        RunResult(
                            status="skipped",
                            reason="no_loop_instrumented",
                            command=[],
                            returncode=None,
                            stdout="",
                            stderr="",
                        ),
                    )
                    summary_rows.append(
                        {
                            "csv": str(csv_path),
                            "row_index": row_index,
                            "target_file": target_file,
                            "source_file": str(source_path),
                            "status": "skipped",
                            "reason": "no_loop_instrumented",
                            "log_file": str(log_path),
                        }
                    )
                    continue

                instrumented_path.write_text(instrumented_code, encoding="utf-8")
                run_result = run_seahorn(
                    instrumented_path,
                    args.docker_image,
                    args.timeout,
                    docker_name=docker_name,
                )

                if not args.keep_instrumented:
                    instrumented_path.unlink(missing_ok=True)

                header["status"] = run_result.status
                header["reason"] = run_result.reason
                write_log(
                    log_path,
                    header,
                    invariants_raw,
                    cleaned_invs,
                    [{"invariant": inv, "reason": reason} for inv, reason in dropped_invs],
                    scan_report,
                    instrument_report,
                    run_result,
                )

                summary_rows.append(
                    {
                        "csv": str(csv_path),
                        "row_index": row_index,
                        "target_file": target_file,
                        "source_file": str(source_path),
                        "status": run_result.status,
                        "reason": run_result.reason,
                        "log_file": str(log_path),
                    }
                )

    if summary_rows:
        fieldnames = list(summary_rows[0].keys())
        with open(summary_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f"[OK] Summary written to {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
